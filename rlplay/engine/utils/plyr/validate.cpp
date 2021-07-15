#include <Python.h>
#include <validate.h>


int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyDict_Size(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        if(stack != NULL)
            stack->push_back(PyLong_FromSsize_t(j+1));

        PyObject *key, *value, *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyDict_Check(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", Py_TYPE(obj)->tp_name));

            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        Py_ssize_t pos = 0;
        while (PyDict_Next(main, &pos, &key, &value)) {
            if(!PyDict_Contains(obj, key)) {
                if(stack != NULL) {
                    Py_INCREF(key);
                    stack->push_back(key);
                }

                PyErr_SetObject(PyExc_KeyError, key);
                return 0;
            }
        }

        if(numel != PyDict_Size(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", "dict size mismatch"));

            PyErr_SetString(PyExc_RuntimeError, "dict size mismatch");
            return 0;
        }

        if(stack != NULL) {
            Py_DECREF(stack->back());
            stack->pop_back();
        }
    }

    return 1;
}


int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyTuple_GET_SIZE(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        if(stack != NULL)
            stack->push_back(PyLong_FromSsize_t(j+1));

        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyTuple_Check(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", Py_TYPE(obj)->tp_name));

            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        if(numel != PyTuple_GET_SIZE(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", "tuple length mismatch"));

            PyErr_SetString(PyExc_RuntimeError, "tuple length mismatch");
            return 0;
        }

        if(stack != NULL) {
            Py_DECREF(stack->back());
            stack->pop_back();
        }
    }

    return 1;
}


int _validate_list(PyObject *main, PyObject *rest, objectstack *stack)
{
    Py_ssize_t numel = PyList_GET_SIZE(main);
    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        if(stack != NULL)
            stack->push_back(PyLong_FromSsize_t(j+1));

        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyList_Check(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", Py_TYPE(obj)->tp_name));

            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        if(numel != PyList_GET_SIZE(obj)) {
            if(stack != NULL)
                stack->push_back(Py_BuildValue("s", "list length mismatch"));

            PyErr_SetString(PyExc_RuntimeError, "list length mismatch");
            return 0;
        }

        if(stack != NULL) {
            Py_DECREF(stack->back());
            stack->pop_back();
        }
    }

    return 1;
}


static PyObject* PyList_fromVector(objectstack &stack)
{
    // steals references from the std::vector
    PyObject *list = PyList_New(stack.size());
    if(list == NULL)
        return NULL;

    for(Py_ssize_t j = 0; j < stack.size(); ++j) {
        PyObject *item = stack[j];
        if(item == NULL) {
            Py_DECREF(list);
            return NULL;
        }

        PyList_SET_ITEM(list, j, item);
    }

    stack.clear();

    return list;
}


int _validate(PyObject *main, PyObject *rest, objectstack &stack)
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

    // the vector is ajust a temporary proxy for a list, and thus steals references
    std::vector<PyObject *> stack = {};

    // dfs through the structures: updates stack and set exceptions in case of an emergency
    _validate(main, rest, stack);

    Py_DECREF(rest);

    PyErr_Clear();

    return PyList_fromVector(stack);
}
