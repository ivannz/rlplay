#include <Python.h>
// https://edcjones.tripod.com/refcount.html

static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                        bool const safe, bool const star, PyObject *kwargs);

int _validate_dict(PyObject *main, PyObject *rest)
{
    Py_ssize_t len = PyDict_Size(main);

    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyDict_Check(obj)) {
            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        if(len != PyDict_Size(obj)) {
            PyErr_SetString(PyExc_RuntimeError, "dict size mismatch");
            return 0;
        }

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(main, &pos, &key, &value)) {
            if(!PyDict_Contains(obj, key)) {
                PyErr_SetObject(PyExc_KeyError, key);
                return 0;
            }
        }
    }

    return 1;
}

static PyObject* _apply_dict(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs)
{
    PyObject *output = PyDict_New(), *result = NULL;
    if(output == NULL) return NULL;

    Py_ssize_t j, p = 0, len = PyTuple_GET_SIZE(rest);
    PyObject *key, *main_, *item_, *rest_ = PyTuple_New(len);
    while (PyDict_Next(main, &p, &key, &main_)) {
        for(j = 0; j < len; j++) {
            item_ = PyDict_GetItem(PyTuple_GET_ITEM(rest, j), key);

            // a tuple assumes ownership of, or 'steals', the reference, owned
            // by a dict from `rest`, so we incref it for protection. It also
            // decrefs `any item already in the tuple at the affected position
            // (if non NULL).`
            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        // `result` is a new object, for which we are now responsible
        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        // dict's setitem does an incref of its own (both value and the key),
        // which is why `_apply_dict` logic appears different from `_tuple`
        PyDict_SetItem(output, key, result);

        // decref the result, so that only `output` owns a ref
        Py_DECREF(result);
    }

    // decrefing a tuple also decrefs all its items
    Py_DECREF(rest_);

    return output;
}

int _validate_tuple(PyObject *main, PyObject *rest)
{
    Py_ssize_t len = PyTuple_GET_SIZE(main);

    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyTuple_Check(obj)) {
            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        if(len != PyTuple_GET_SIZE(obj)) {
            PyErr_SetString(PyExc_RuntimeError, "tuple length mismatch");
            return 0;
        }
    }

    return 1;
}

static PyObject* _apply_tuple(PyObject *callable, PyObject *main, PyObject *rest,
                              bool const safe, bool const star, PyObject *kwargs)
{
    Py_ssize_t numel = PyTuple_GET_SIZE(main);
    PyObject *output = PyTuple_New(numel), *result = NULL;
    if(output == NULL) return NULL;

    Py_ssize_t j, p, len = PyTuple_GET_SIZE(rest);
    PyObject *main_, *item_,  *rest_ = PyTuple_New(len);
    for(p = 0; p < numel; p++) {
        main_ = PyTuple_GET_ITEM(main, p);
        for(j = 0; j < len; j++) {
            item_ = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rest, j), p);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        PyTuple_SET_ITEM(output, p, result);
    }

    Py_DECREF(rest_);

    if(PyTuple_CheckExact(main))
        return output;

    PyObject *namedtuple = Py_TYPE(main)->tp_new(Py_TYPE(main), output, NULL);
    Py_DECREF(output);

    return namedtuple;
}

int _validate_list(PyObject *main, PyObject *rest)
{
    Py_ssize_t len = PyList_GET_SIZE(main);

    for(Py_ssize_t j = 0; j < PyTuple_GET_SIZE(rest); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(rest, j);

        if(!PyList_Check(obj)) {
            PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
            return 0;
        }

        if(len != PyList_GET_SIZE(obj)) {
            PyErr_SetString(PyExc_RuntimeError, "list length mismatch");
            return 0;
        }
    }

    return 1;
}

static PyObject* _apply_list(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs)
{
    Py_ssize_t numel = PyList_GET_SIZE(main);
    PyObject *output = PyList_New(numel), *result = NULL;
    if(output == NULL) return NULL;

    Py_ssize_t j, p, len = PyTuple_GET_SIZE(rest);
    PyObject *main_, *item_,  *rest_ = PyTuple_New(len);
    for(p = 0; p < numel; p++) {
        main_ = PyList_GET_ITEM(main, p);
        for(j = 0; j < len; j++) {
            item_ = PyList_GET_ITEM(PyTuple_GET_ITEM(rest, j), p);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        PyList_SET_ITEM(output, p, result);
    }

    Py_DECREF(rest_);

    return output;
}

static PyObject* _apply_mapping(PyObject *callable, PyObject *main, PyObject *rest,
                                bool const safe, bool const star, PyObject *kwargs)
{
    // XXX it's unlikely that we will ever use this branch, because as docs
    // it is impossible to know the type of keys of a mapping at runtime,
    //  hence lists, tuples, dicts and any objects with `__getitem__` are
    //  mappings according to `PyMapping_Check`.
    PyObject *output = PyDict_New(), *result = Py_None;
    if(output == NULL) return NULL;
    Py_INCREF(result);

    Py_ssize_t j, p, len = PyTuple_GET_SIZE(rest);
    PyObject *key, *main_, *item_, *rest_ = PyTuple_New(len);

    PyObject *items = PyMapping_Items(main);
    if(items == NULL) return NULL;

    Py_ssize_t numel = PyList_GET_SIZE(items);
    for(p = 0; p < numel; p++) {
        item_ = PyList_GET_ITEM(items, p);
        key = PyTuple_GET_ITEM(item_, 0);
        main_ = PyTuple_GET_ITEM(item_, 1);

        for(j = 0; j < len; j++) {
            item_ = PyObject_GetItem(PyTuple_GET_ITEM(rest, j), key);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        Py_DECREF(result);

        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) break;

        PyDict_SetItem(output, key, result);
    }

    Py_DECREF(items);
    Py_DECREF(rest_);

    if(result == NULL) return NULL;
    Py_DECREF(result);

    return output;
}

static PyObject* _apply_base(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const star, PyObject *kwargs)
{
    PyObject *output;

    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    PyObject *item_, *args = PyTuple_New(1+len);
    if(args == NULL) return NULL;

    Py_INCREF(main);
    PyTuple_SET_ITEM(args, 0, main);
    for(Py_ssize_t j = 0; j < len; j++) {
        item_ = PyTuple_GET_ITEM(rest, j);

        Py_INCREF(item_);
        PyTuple_SET_ITEM(args, j + 1, item_);
    }

    if (star) {
        output = PyObject_Call(callable, args, kwargs);
        Py_DECREF(args);

    } else {
        PyObject *one = PyTuple_New(1);
        PyTuple_SET_ITEM(one, 0, args);

        output = PyObject_Call(callable, one, kwargs);
        Py_DECREF(one);
    }

    return output;
}

static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                        bool const safe, bool const star, PyObject *kwargs)
{
    PyObject *result;

    if(PyDict_Check(main)) {
        if(safe)
            if(!_validate_dict(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_dict(callable, main, rest, safe, star, kwargs);
        Py_LeaveRecursiveCall();

    } else if(PyTuple_Check(main)) {
        if(safe)
            if(!_validate_tuple(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_tuple(callable, main, rest, safe, star, kwargs);
        Py_LeaveRecursiveCall();

    } else if(PyList_Check(main)) {
        if(safe)
            if(!_validate_list(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_list(callable, main, rest, safe, star, kwargs);
        Py_LeaveRecursiveCall();

    } else {
        result = _apply_base(callable, main, rest, star, kwargs);

    }

    return result;
}

static PyObject* apply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // from the url at the top: {API 1.2.1} the call mechanism guarantees
    //  to hold a reference to every argument for the duration of the call.
    int safe = 1, star = 1;
    PyObject *callable = NULL, *main = NULL;

    //handle `apply(fn, main, *rest, *, ...)`
    Py_ssize_t len = PyTuple_GET_SIZE(args);
    PyObject *first = PyTuple_GetSlice(args, 0, 2);
    PyObject *rest = PyTuple_GetSlice(args, 2, len);

    int parsed = PyArg_ParseTuple(first, "OO|:apply", &callable, &main);
    Py_DECREF(first);
    if (!parsed) return NULL;

    if(!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "The first argument must be a callable.");
        return NULL;
    }

    //handle `apply(..., *, _star, _safe, **kwargs)`
    if (kwargs) {
        static char *kwlist[] = {"_safe", "_star", NULL};

        PyObject *empty = PyTuple_New(0);
        if (empty == NULL) return NULL;

        PyObject* own = PyDict_New();
        if (empty == NULL) {
            Py_DECREF(empty);
            return NULL;
        }

        for(int p = 0; kwlist[p] != NULL; p++) {
            PyObject* arg = PyDict_GetItemString(kwargs, kwlist[p]);
            if (arg == NULL) continue;

            PyDict_SetItemString(own, kwlist[p], arg);
            PyDict_DelItemString(kwargs, kwlist[p]);
        }

        int parsed = PyArg_ParseTupleAndKeywords(
                empty, own, "|$pp:apply", kwlist, &safe, &star);

        Py_DECREF(empty);
        Py_DECREF(own);
        if (!parsed) return NULL;
    }

    PyObject *result = _apply(callable, main, rest, safe, star, kwargs);
    Py_DECREF(rest);

    return result;
}

static PyMethodDef modapply_methods[] = {
    {
        "apply",
        (PyCFunction) apply,
        METH_VARARGS | METH_KEYWORDS,
        "Pure C implementation of apply, with optional safety checks.",
    }, {
        NULL,
        NULL,
        0,
        NULL,
    }
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_apply",
        NULL,
        -1,
        modapply_methods,
};


PyMODINIT_FUNC
PyInit__apply(void)
{
    return PyModule_Create(&moduledef);
}
