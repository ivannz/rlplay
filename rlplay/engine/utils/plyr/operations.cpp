#include <Python.h>


PyObject* getitem(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"", "index", NULL};

    PyObject *object = NULL, *index = NULL;
    int parsed = PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:getitem",
                                             kwlist, &object, &index);
    if(!parsed)
        return NULL;

    return PyObject_GetItem(object, index);
}


PyObject* setitem(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"", "", "index", NULL};

    PyObject *object = NULL, *value = NULL, *index = NULL;
    int parsed = PyArg_ParseTupleAndKeywords(
        args, kwargs, "OO|$O:setitem", kwlist, &object, &value, &index);
    if(!parsed)
        return NULL;

    if (-1 == PyObject_SetItem(object, index, value))
        return NULL;

    Py_RETURN_NONE;
}


PyObject* is_sequence(PyObject *self, PyObject *object)
{
    if(PySequence_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}


PyObject* is_mapping(PyObject *self, PyObject *object)
{
    if(PyMapping_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}


const PyMethodDef def_getitem = {
    "getitem",
    (PyCFunction) getitem,
    METH_VARARGS | METH_KEYWORDS,
    "getitem(object, *, index) returns object[index]",
};


const PyMethodDef def_setitem = {
    "setitem",
    (PyCFunction) setitem,
    METH_VARARGS | METH_KEYWORDS,
    "setitem(object, value, *, index) does object[index] = value",
};


const PyMethodDef def_is_sequence = {
    "is_sequence",
    (PyCFunction) is_sequence,
    METH_O,
    NULL,
};


const PyMethodDef def_is_mapping = {
    "is_mapping",
    (PyCFunction) is_mapping,
    METH_O,
    NULL,
};
