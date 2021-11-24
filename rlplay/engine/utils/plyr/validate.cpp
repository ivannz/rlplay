#include <Python.h>
#include <validate.h>


PyDoc_STRVAR(
    __doc__,
    "\n"
    "validate(*objects)\n"
    "\n"
    "Validate the structure of the nested objects (see `apply` and caveats).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "result : list\n"
    "    An empty list, if the objects have IDENTICAL structure (nesting and\n"
    "    container types). Otherwise, contains the error in its last element\n"
    "    preceded by the index/key path within the nesting structure. The error\n"
    "    is a tuple with three elements: the index of the object in the arguments,\n"
    "    the type of the raised exception, and the value of the exception.\n"
    "\n"
);


int _raise_TypeError(Py_ssize_t index, PyObject *main, PyObject *obj, objectstack *stack)
{
    char error[160];
    PyOS_snprintf(error, 160, "Expected '%s', got '%s'",
                  Py_TYPE(main)->tp_name, Py_TYPE(obj)->tp_name);

    if(stack != NULL) {
        stack->push_back(Py_BuildValue("(nOs)", index, PyExc_TypeError, error));

    } else {
        PyErr_SetString(PyExc_TypeError, error);

    }

    return 0;
}


int _raise_SizeError(Py_ssize_t index, PyObject *main, objectstack *stack)
{
    char error[160];
    PyOS_snprintf(error, 160, "'%s' size mismatch", Py_TYPE(main)->tp_name);

    if(stack != NULL) {
        stack->push_back(Py_BuildValue("(nOs)", index, PyExc_RuntimeError, error));

    } else {
        PyErr_SetString(PyExc_RuntimeError, error);

    }

    return 0;
}


int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyDict_Size(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *key, *value, *obj = PyTuple_GET_ITEM(rest, j);

        if(!Py_IS_TYPE(obj, Py_TYPE(main)))
            return _raise_TypeError(j+1, main, obj, stack);

        if(numel != PyDict_Size(obj))
            return _raise_SizeError(j+1, main, stack);

        Py_ssize_t pos = 0;
        while (PyDict_Next(main, &pos, &key, &value)) {
            if(!PyDict_Contains(obj, key)) {
                // no need for incref on the key, since both `buildvalue`
                // and `setobject` incref the value.
                if(stack != NULL) {
                    stack->push_back(Py_BuildValue(
                        "(nOO)", j+1, PyExc_KeyError, key));

                } else {
                    PyErr_SetObject(PyExc_KeyError, key);

                }

                return 0;
            }
        }
    }

    return 1;
}


int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyTuple_GET_SIZE(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!Py_IS_TYPE(obj, Py_TYPE(main)))
            return _raise_TypeError(j+1, main, obj, stack);

        if(numel != PyTuple_GET_SIZE(obj))
            return _raise_SizeError(j+1, main, stack);
    }

    return 1;
}


int _validate_list(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyList_GET_SIZE(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!Py_IS_TYPE(obj, Py_TYPE(main)))
            return _raise_TypeError(j+1, main, obj, stack);

        if(numel != PyList_GET_SIZE(obj))
            return _raise_SizeError(j+1, main, stack);
    }

    return 1;
}


static PyObject* PyList_fromVector(objectstack &stack)
{
    PyObject *list = PyList_New(stack.size());
    if(list == NULL) {
        // could not allocate new list: decref stolen refs
        for(Py_ssize_t j = 0; j < stack.size(); ++j)
            Py_XDECREF(stack[j]);

    } else {
        // transfer the stolen ownership from std::vector to the list
        for(Py_ssize_t j = 0; j < stack.size(); ++j)
            PyList_SET_ITEM(list, j, stack[j]);

    }

    stack.clear();

    return list;
}


static int _validate(PyObject *main, PyObject *rest, objectstack &stack)
{
    int result;

    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    if (len == 0)
        return 1;

    PyObject *key, *rest_ = PyTuple_New(len);
    if(rest_ == NULL)
        return 0;

    PyObject *main_, *item_;
    if(PyDict_Check(main)) {
        if(!_validate_dict(main, rest, &stack))
            return 0;

        // for each key in the main dict
        Py_ssize_t pos = 0;
        while (PyDict_Next(main, &pos, &key, &main_)) {
            Py_INCREF(key);
            stack.push_back(key);

            for(Py_ssize_t j = 0; j < len; j++) {
                item_ = PyDict_GetItem(PyTuple_GET_ITEM(rest, j), key);

                Py_INCREF(item_);
                // XXX we're fine here with `PyTuple_SetItem`-s extra safety
                PyTuple_SetItem(rest_, j, item_);
            }

            if(Py_EnterRecursiveCall("")) return 0;
            result = _validate(main_, rest_, stack);
            Py_LeaveRecursiveCall();

            if(!result) {
                Py_DECREF(rest_);
                return 0;
            }

            stack.pop_back();
            Py_DECREF(key);
        }

    } else if(PyTuple_Check(main)) {
        if(!_validate_tuple(main, rest, &stack))
            return 0;

        for(Py_ssize_t pos = 0; pos < PyTuple_GET_SIZE(main); pos++) {
            key = PyLong_FromSsize_t(pos);
            if(key == NULL) {
                Py_DECREF(rest_);
                return 0;
            }

            stack.push_back(key);

            main_ = PyTuple_GET_ITEM(main, pos);
            for(Py_ssize_t j = 0; j < len; j++) {
                item_ = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

                Py_INCREF(item_);
                PyTuple_SetItem(rest_, j, item_);
            }

            if(Py_EnterRecursiveCall("")) return 0;
            result = _validate(main_, rest_, stack);
            Py_LeaveRecursiveCall();

            if(!result) {
                Py_DECREF(rest_);
                return 0;
            }

            stack.pop_back();
            Py_DECREF(key);
        }

    } else if(PyList_Check(main)) {
        if(!_validate_list(main, rest, &stack))
            return 0;

        for(Py_ssize_t pos = 0; pos < PyList_GET_SIZE(main); pos++) {
            key = PyLong_FromSsize_t(pos);
            if(key == NULL) {
                Py_DECREF(rest_);
                return 0;
            }

            stack.push_back(key);

            main_ = PyList_GET_ITEM(main, pos);
            for(Py_ssize_t j = 0; j < len; j++) {
                item_ = PyList_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

                Py_INCREF(item_);
                PyTuple_SetItem(rest_, j, item_);
            }

            if(Py_EnterRecursiveCall("")) return 0;
            result = _validate(main_, rest_, stack);
            Py_LeaveRecursiveCall();

            if(!result) {
                Py_DECREF(rest_);
                return 0;
            }

            stack.pop_back();
            Py_DECREF(key);
        }

    }

    // decrefing a tuple also decrefs all its items
    Py_DECREF(rest_);

    return 1;
}


PyObject* validate(PyObject *self, PyObject *args)
{
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    if(len == 1)
        return PyList_New(0);

    PyObject *main = NULL;

    PyObject *first = PyTuple_GetSlice(args, 0, 1);
    int parsed = PyArg_ParseTuple(first, "O|:validate", &main);
    Py_DECREF(first);

    if (!parsed)
        return NULL;

    PyObject *rest = PyTuple_GetSlice(args, 1, len);
    if (rest == NULL)
        return NULL;

    // the vector is adjust a temporary proxy for a list, and thus
    //  steals references
    std::vector<PyObject *> stack = {};

    // dfs through the structures: updates stack and set exceptions
    //  in case of an emergency
    _validate(main, rest, stack);
    Py_DECREF(rest);

    if(PyErr_Occurred() != NULL) {
        for(Py_ssize_t j = 0; j < stack.size(); ++j)
            Py_XDECREF(stack[j]);

        stack.clear();
        return NULL;
    }

    return PyList_fromVector(stack);
}


const PyMethodDef def_validate = {
    "validate",
    (PyCFunction) validate,
    METH_VARARGS,
    __doc__,
};
