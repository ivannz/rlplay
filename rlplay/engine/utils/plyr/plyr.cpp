#include <Python.h>

#include <apply.h>
#include <validate.h>
#include <operations.h>

// apply functions with preset _safe and _star kwargs
// [ts][u_]apply -- t/s tuple or star args, u/_ unsafe or safe
static PyObject* suply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 0, 1, kwargs);
    Py_DECREF(rest);

    return result;
}


static PyObject* tuply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 0, 0, kwargs);
    Py_DECREF(rest);

    return result;
}


static PyObject* s_ply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 1, 1, kwargs);
    Py_DECREF(rest);

    return result;
}


static PyObject* t_ply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 1, 0, kwargs);
    Py_DECREF(rest);

    return result;
}


static PyMethodDef modplyr_methods[] = {
    def_apply, {
        "suply",
        (PyCFunction) suply,
        METH_VARARGS | METH_KEYWORDS,
        "Star-apply without safety checks (use at your own risk).",
    }, {
        "tuply",
        (PyCFunction) tuply,
        METH_VARARGS | METH_KEYWORDS,
        "Tuple-apply without safety checks (use at your own risk).",
    }, {
        "s_ply",
        (PyCFunction) s_ply,
        METH_VARARGS | METH_KEYWORDS,
        "Star-apply with safety checks.",
    }, {
        "t_ply",
        (PyCFunction) t_ply,
        METH_VARARGS | METH_KEYWORDS,
        "Tuple-apply with safety checks.",
    }, {
        "getitem",
        (PyCFunction) getitem,
        METH_VARARGS | METH_KEYWORDS,
        "getitem(object, *, index) returns object[index]",
    }, {
        "setitem",
        (PyCFunction) setitem,
        METH_VARARGS | METH_KEYWORDS,
        "setitem(object, value, *, index) does object[index] = value",
    }, {
        "is_sequence",
        (PyCFunction) is_sequence,
        METH_O,
        NULL,
    }, {
        "is_mapping",
        (PyCFunction) is_mapping,
        METH_O,
        NULL,
    }, {
        "validate",
        (PyCFunction) validate,
        METH_VARARGS,
        "validate(*objects) validates the structure of the nested objects",
    }, {
        NULL,
        NULL,
        0,
        NULL,
    }
};


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "plyr",
        NULL,
        -1,
        modplyr_methods,
};


PyMODINIT_FUNC
PyInit_plyr(void)
{
    return PyModule_Create(&moduledef);
}
