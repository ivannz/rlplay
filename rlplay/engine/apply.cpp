#include <Python.h>

static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest, bool const safe);

static PyObject* _apply_dict(PyObject *callable, PyObject *main, PyObject *rest, bool const safe)
{
    PyObject *output = PyDict_New();
    if(output == NULL) return NULL;

    Py_ssize_t j, p = 0, len = PyTuple_Size(rest);
    PyObject *key, *main_, *dict_, *item_, *rest_ = PyTuple_New(len);
    while (PyDict_Next(main, &p, &key, &main_)) {
        for(j = 0; j < len; j++) {
            item_ = PyDict_GetItem(PyTuple_GET_ITEM(rest, j), key);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        PyObject *result = _apply(callable, main_, rest_, safe);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }
        PyDict_SetItem(output, key, result);
        Py_DECREF(result);
    }

    Py_DECREF(rest_);

    return output;
}

static PyObject* _apply_tuple(PyObject *callable, PyObject *main, PyObject *rest, bool const safe)
{
    Py_ssize_t numel = PyTuple_Size(main);
    PyObject *output = PyTuple_New(numel);
    if(output == NULL) return NULL;

    Py_ssize_t j, p, len = PyTuple_Size(rest);
    PyObject *main_, *item_,  *rest_ = PyTuple_New(len);
    for(p = 0; p < numel; p++) {
        main_ = PyTuple_GET_ITEM(main, p);
        for(j = 0; j < len; j++) {
            item_ = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rest, j), p);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        PyObject *result = _apply(callable, main_, rest_, safe);
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

    PyObject *result = Py_TYPE(main)->tp_new(Py_TYPE(main), output, NULL);
    Py_DECREF(output);

    return result;
}

static PyObject* _apply_list(PyObject *callable, PyObject *main, PyObject *rest, bool const safe)
{
    Py_ssize_t numel = PyList_Size(main);
    PyObject *output = PyList_New(numel);
    if(output == NULL) return NULL;

    Py_ssize_t j, p, len = PyTuple_Size(rest);
    PyObject *main_, *item_,  *rest_ = PyTuple_New(len);
    for(p = 0; p < numel; p++) {
        main_ = PyList_GET_ITEM(main, p);
        for(j = 0; j < len; j++) {
            item_ = PyList_GET_ITEM(PyTuple_GET_ITEM(rest, j), p);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        PyObject *result = _apply(callable, main_, rest_, safe);
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

static PyObject* _apply_base(PyObject *callable, PyObject *main, PyObject *rest)
{
    Py_ssize_t len = PyTuple_Size(rest);
    PyObject *item_, *args = PyTuple_New(1+len);
    if(args == NULL) return NULL;


    Py_INCREF(main);
    PyTuple_SetItem(args, 0, main);
    for(Py_ssize_t j = 0; j < len; j++) {
        item_ = PyTuple_GET_ITEM(rest, j);

        Py_INCREF(item_);
        PyTuple_SetItem(args, j + 1, item_);
    }

    PyObject *output = PyObject_Call(callable, args, NULL);
    Py_DECREF(args);

    return output;
}

static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest, bool const safe)
{
    if(PyDict_Check(main)) {
        if(safe) {
            Py_ssize_t len = PyDict_Size(main);

            for(Py_ssize_t j = 0; j < PyTuple_Size(rest); ++j) {
                PyObject *obj = PyTuple_GET_ITEM(rest, j);

                if(!PyDict_Check(obj)) {
                    PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
                    return NULL;
                }

                if(len != PyDict_Size(obj)) {
                    PyErr_SetString(PyExc_RuntimeError, "dict size mismatch");
                    return NULL;
                }

                PyObject *key, *value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(main, &pos, &key, &value)) {
                    if(!PyDict_Contains(obj, key)) {
                        PyErr_SetObject(PyExc_KeyError, key);
                        return NULL;
                    }
                }
            }
        }

        return _apply_dict(callable, main, rest, safe);

    } else if(PyTuple_Check(main)) {
        if(safe) {
            Py_ssize_t len = PyTuple_Size(main);

            for(Py_ssize_t j = 0; j < PyTuple_Size(rest); ++j) {
                PyObject *obj = PyTuple_GET_ITEM(rest, j);

                if(!PyTuple_Check(obj)) {
                    PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
                    return NULL;
                }

                if(len != PyTuple_Size(obj)) {
                    PyErr_SetString(PyExc_RuntimeError, "tuple length mismatch");
                    return NULL;
                }
            }
        }

        return _apply_tuple(callable, main, rest, safe);

    } else if(PyList_Check(main)) {
        if(safe) {
            Py_ssize_t len = PyList_Size(main);

            for(Py_ssize_t j = 0; j < PyTuple_Size(rest); ++j) {
                PyObject *obj = PyTuple_GET_ITEM(rest, j);

                if(!PyList_Check(obj)) {
                    PyErr_SetString(PyExc_TypeError, Py_TYPE(obj)->tp_name);
                    return NULL;
                }

                if(len != PyList_Size(obj)) {
                    PyErr_SetString(PyExc_RuntimeError, "list length mismatch");
                    return NULL;
                }
            }
        }

        return _apply_list(callable, main, rest, safe);
    }

    return _apply_base(callable, main, rest);
}

static PyObject* apply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int safe = 1;

    if (kwargs) {
        PyObject *empty = PyTuple_New(0);
        if (empty == NULL) return NULL;

        static char *kwlist[] = {"safe", NULL};
        int parsed = PyArg_ParseTupleAndKeywords(
                empty, kwargs, "|$p:apply", kwlist, &safe);

        Py_DECREF(empty);
        if (!parsed) return NULL;
    }

    PyObject *result = NULL;
    Py_ssize_t len = PyTuple_Size(args);
    PyObject *callable = PyTuple_GetItem(args, 0);
    // Py_INCREF(callable);

    if(!PyCallable_Check(callable)) {
        PyErr_SetObject(PyExc_TypeError, callable);
        return NULL;
    }

    PyObject *main = PyTuple_GetItem(args, 1);
    PyObject *rest = PyTuple_GetSlice(args, 2, len);

    result = _apply(callable, main, rest, safe);
    Py_DECREF(rest);

    return result;
}


static PyMethodDef modapply_methods[] = {
    {
        "apply",
        (PyCFunction) apply,
        METH_VARARGS | METH_KEYWORDS,
        "Pure C implementation of apply, with optional safety checks.",
    }, {NULL, NULL, 0, NULL,}
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
