#include <Python.h>
#include <operations.h>


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


const PyMethodDef def_getitem = {
    "getitem",
    (PyCFunction) getitem,
    METH_VARARGS | METH_KEYWORDS,
    PyDoc_STR(
        "getitem(object, *, index) returns object[index]"
    ),
};


PyObject* xgetitem(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // quick check for None
    PyObject *object = PyTuple_GetItem(args, 0);
    // https://docs.python.org/3/c-api/none.html
    if(object == Py_None)
        Py_RETURN_NONE;

    return getitem(self, args, kwargs);
}


const PyMethodDef def_xgetitem = {
    "xgetitem",
    (PyCFunction) xgetitem,
    METH_VARARGS | METH_KEYWORDS,
    PyDoc_STR(
        "`xgetitem` -- the same as `getitem`, but allows None through"
    ),
};


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


const PyMethodDef def_setitem = {
    "setitem",
    (PyCFunction) setitem,
    METH_VARARGS | METH_KEYWORDS,
    PyDoc_STR(
        "setitem(object, value, *, index) does object[index] = value"
    ),
};


PyObject* xsetitem(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // quick check for None
    PyObject *object = PyTuple_GetItem(args, 0);
    // https://docs.python.org/3/c-api/none.html
    if(object == Py_None)
        Py_RETURN_NONE;

    return setitem(self, args, kwargs);
}


const PyMethodDef def_xsetitem = {
    "xsetitem",
    (PyCFunction) xsetitem,
    METH_VARARGS | METH_KEYWORDS,
    PyDoc_STR(
        "`xsetitem` -- the same as `setitem`, but allows None through"
    ),
};


PyObject* is_sequence(PyObject *self, PyObject *object)
{
    if(PySequence_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}


const PyMethodDef def_is_sequence = {
    "is_sequence",
    (PyCFunction) is_sequence,
    METH_O,
    NULL,
};


PyObject* is_mapping(PyObject *self, PyObject *object)
{
    if(PyMapping_Check(object)) {
        Py_RETURN_TRUE;

    } else {
        Py_RETURN_FALSE;

    }
}


const PyMethodDef def_is_mapping = {
    "is_mapping",
    (PyCFunction) is_mapping,
    METH_O,
    NULL,
};


PyObject* dict_getrefs(PyObject *self, PyObject *dict)
{
    if(PyDict_Check(dict)) {
        PyObject *tuple = PyTuple_New(PyDict_Size(dict));
        if(tuple == NULL)
            return NULL;

        PyObject *key, *val;
        Py_ssize_t pos = 0, j = 0;
        while(PyDict_Next(dict, &pos, &key, &val))
            PyTuple_SET_ITEM(
                tuple, j++,
                Py_BuildValue("nn", Py_REFCNT(key), Py_REFCNT(val)));

        return tuple;
    }

    Py_RETURN_NONE;
}


PyDoc_STRVAR(
    dict_getrefs__doc__,
    "\n"
    "class Foo:\n"
    "    \"\"\"hashable mockup object\"\"\"\n"
    "    def __hash__(self):\n"
    "        return 0xDEADC0DE\n"
    "\n"
    "\n"
    "from sys import getrefcount\n"
    "\n"
    "# `getrefcount` is \"generally one higher than you might expect\"\n"
    "print(getrefcount(Foo()))  # 1\n"
    "\n"
    "f = Foo()  # 1\n"
    "print(getrefcount(f))  # 1+1\n"
    "\n"
    "print(dict_getrefs({f: f}))  # 3\n"
    "print(getrefcount(f))  # 1+1\n"
    "\n"
    "d1 = {f: f}  # 3\n"
    "print(dict_getrefs(d1))  # 3\n"
    "print(getrefcount(f))  # 3+1\n"
    "\n"
    "\n"
    "d2 = dict_clone(d1)  # 5 key and item are increfed by `dict-setitem`\n"
    "print(dict_getrefs(d1))  # 5\n"
    "print(getrefcount(f))  # 5+1\n"
);


const PyMethodDef def_dict_getrefs = {
    "dict_getrefs",
    (PyCFunction) dict_getrefs,
    METH_O,
    dict_getrefs__doc__,
};


PyObject* dict_clone(PyObject *self, PyObject *dict)
{
    if(!PyDict_Check(dict)) {
        PyErr_SetNone(PyExc_TypeError);

        return NULL;
    }

    PyObject *key, *result;

    PyObject *output = PyDict_New();
    if(output == NULL)
        return NULL;

    Py_ssize_t pos = 0;
    while(PyDict_Next(dict, &pos, &key, &result)) {
        PyDict_SetItem(output, key, result);
    }

    return output;
}


const PyMethodDef def_dict_clone = {
    "dict_clone",
    (PyCFunction) dict_clone,
    METH_O,
    NULL,
};
