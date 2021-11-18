#include <Python.h>

#include <apply.h>
#include <validate.h>
#include <operations.h>


PyDoc_STRVAR(
    __doc__,
    "\n"
    "Streamlined operations on built-in nested containers of objects.\n"
    "\n"
    "see `plyr.apply`.\n"
);

// apply functions with preset _safe and _star kwargs
// [ts][u_]apply -- t/s tuple or star args, u/_ unsafe or safe
static PyObject* suply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 0, 1, kwargs, NULL);
    Py_DECREF(rest);

    return result;
}


static PyObject* tuply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 0, 0, kwargs, NULL);
    Py_DECREF(rest);

    return result;
}


static PyObject* s_ply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 1, 1, kwargs, NULL);
    Py_DECREF(rest);

    return result;
}


static PyObject* t_ply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable = NULL, *main = NULL, *rest = NULL;
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    PyObject *result = _apply(callable, main, rest, 1, 0, kwargs, NULL);
    Py_DECREF(rest);

    return result;
}


static PyMethodDef modplyr_methods[] = {
    def_apply,
    def_validate,
    {
        "suply",
        (PyCFunction) suply,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR(
            "Star-apply without safety checks (use at your own risk)."
        ),
    }, {
        "tuply",
        (PyCFunction) tuply,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR(
            "Tuple-apply without safety checks (use at your own risk)."
        ),
    }, {
        "s_ply",
        (PyCFunction) s_ply,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR(
            "Star-apply with safety checks."
        ),
    }, {
        "t_ply",
        (PyCFunction) t_ply,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR(
            "Tuple-apply with safety checks."
        ),
    },
    def_getitem,
    def_setitem,

    def_xgetitem,
    def_xsetitem,

    // XXX debug functions
    def_is_sequence,
    def_is_mapping,
    def_dict_getrefs,
    def_dict_clone,
    {
        NULL,
        NULL,
        0,
        NULL,
    }
};


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "plyr",
        __doc__,
        -1,
        modplyr_methods,
};


PyMODINIT_FUNC
PyInit_plyr(void)
{
    return PyModule_Create(&moduledef);
}
