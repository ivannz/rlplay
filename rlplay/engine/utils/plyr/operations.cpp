#include <Python.h>


static PyObject* getitem(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"", "index", NULL};

    PyObject *object = NULL, *index = NULL;
    int parsed = PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:getitem",
                                             kwlist, &object, &index);
    if(!parsed)
        return NULL;

    return PyObject_GetItem(object, index);
}


static PyObject* setitem(PyObject *self, PyObject *args, PyObject *kwargs)
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


static PyObject* is_sequence(PyObject *self, PyObject *object)
{
    if(PySequence_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}


static PyObject* is_mapping(PyObject *self, PyObject *object)
{
    if(PyMapping_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}