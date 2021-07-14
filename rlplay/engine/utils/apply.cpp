#include <Python.h>
#include <vector>
// https://edcjones.tripod.com/refcount.html

typedef std::vector<PyObject *> objectstack;


static const char *__doc__ = "\n"
    "apply(callable, *objects, _safe=True, _star=True, **kwargs)\n"
    "\n"
    "Compute the function using the leaf data of the nested objects as arguments.\n"
    "\n"
    "A `nested object` is either a python object (object, str, numpy array, torch\n"
    "tensor, etc.) or a subclass of one of python's builtin containers (dict,\n"
    "list, or tuple), that consists of other nested objects.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "callable : callable\n"
    "    A callable object to be applied to the leaf data.\n"
    "\n"
    "*objects : nested objects\n"
    "    All remaining positionals to `apply` are assumed to be nested objects,\n"
    "    that supply arguments for the callable from their leaf data.\n"
    "\n"
    "_safe : bool, default=True\n"
    "    Disables structural safety checks when more than one nested object has\n"
    "    been supplied.\n"
    "    SEGFAULTs if the nested objects do not have IDENTICAL STRUCTURE.\n"
    "\n"
    "_star : bool, default=True\n"
    "    Determines how to pass the leaf data to the callable.\n"
    "    If `True` (star-apply), then we call\n"
    "        `callable(d_1, d_2, ..., d_n, **kwargs)`,\n"
    "\n"
    "    otherwise packages the leaf data into a tuple (tuple-apply) and calls\n"
    "        `callable((d_1, d_2, ..., d_n), **kwargs)`\n"
    "\n"
    "    even for `n=1`.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "result : a new nested object\n"
    "    The nested object that contains the values returned by `callable`.\n"
    "    Guaranteed to have IDENTICAL structure as the first nested object\n"
    "    in objects.\n"
    "\n"
    "Details\n"
    "-------\n"
    "For a single container `apply` with `_star=True` is roughly equivalent to\n"
    ">>> def apply(fn, container, **kwargs):\n"
    ">>>     if isinstance(container, dict):\n"
    ">>>         return {k: apply(fn, v, **kwargs)\n"
    ">>>                 for k, v in container.items()}\n"
    ">>>\n"
    ">>>     if isinstance(container, (tuple, list)):\n"
    ">>>         return type(container)([apply(fn, v, **kwargs)\n"
    ">>>                                 for v in container])\n"
    ">>>\n"
    ">>>     return fn(container, **kwargs)\n"
    "\n"
;


static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                        bool const safe, bool const star, PyObject *kwargs);


int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack=NULL)
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


static PyObject* _apply_dict(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs)
{
    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    PyObject *key, *main_, *item_, *rest_ = PyTuple_New(len);
    if(rest_ == NULL)
        return NULL;

    PyObject *output = PyDict_New(), *result = NULL;
    if(output == NULL) {
        Py_DECREF(rest_);
        return NULL;
    }

    Py_ssize_t pos = 0;
    while (PyDict_Next(main, &pos, &key, &main_)) {
        for(Py_ssize_t j = 0; j < len; j++) {
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


int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack=NULL)
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


static PyObject* _apply_tuple(PyObject *callable, PyObject *main, PyObject *rest,
                              bool const safe, bool const star, PyObject *kwargs)
{
    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    PyObject *main_, *item_, *rest_ = PyTuple_New(len);
    if(rest_ == NULL)
        return NULL;

    Py_ssize_t numel = PyTuple_GET_SIZE(main);
    PyObject *output = PyTuple_New(numel), *result = NULL;
    if(output == NULL) {
        Py_DECREF(rest_);
        return NULL;
    }

    for(Py_ssize_t pos = 0; pos < numel; pos++) {
        main_ = PyTuple_GET_ITEM(main, pos);
        for(Py_ssize_t j = 0; j < len; j++) {
            item_ = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        PyTuple_SET_ITEM(output, pos, result);
    }

    Py_DECREF(rest_);

    if(PyTuple_CheckExact(main))
        return output;

    // Preserve namedtuple, devolve others to builtin tuples
    // "isinstance(o, tuple) and hasattr(o, '_fields')" is the corect way.
    //   https://mail.python.org/pipermail//python-ideas/2014-January/024886.html
    //   https://bugs.python.org/issue7796
    if(!PyObject_HasAttrString(main, "_fields"))
        return output;

    PyObject *namedtuple = Py_TYPE(main)->tp_new(Py_TYPE(main), output, NULL);
    Py_DECREF(output);

    return namedtuple;
}


int _validate_list(PyObject *main, PyObject *rest, objectstack *stack=NULL)
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


static PyObject* _apply_list(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs)
{
    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    PyObject *main_, *item_, *rest_ = PyTuple_New(len);
    if(rest_ == NULL)
        return NULL;

    Py_ssize_t numel = PyList_GET_SIZE(main);
    PyObject *output = PyList_New(numel), *result = NULL;
    if(output == NULL) {
        Py_DECREF(rest_);
        return NULL;
    }

    for(Py_ssize_t pos = 0; pos < numel; pos++) {
        main_ = PyList_GET_ITEM(main, pos);
        for(Py_ssize_t j = 0; j < len; j++) {
            item_ = PyList_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        PyList_SET_ITEM(output, pos, result);
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
    Py_ssize_t len = PyTuple_GET_SIZE(rest);
    PyObject *key, *main_, *item_, *rest_ = PyTuple_New(len);
    if(rest_ == NULL)
        return NULL;

    PyObject *output = PyDict_New(), *result = Py_None;
    if(output == NULL) {
        Py_DECREF(rest_);
        return NULL;
    }

    PyObject *items = PyMapping_Items(main);
    if(items == NULL) {
        Py_DECREF(rest_);
        Py_DECREF(output);
        return NULL;
    }

    Py_INCREF(result);

    Py_ssize_t numel = PyList_GET_SIZE(items);
    for(Py_ssize_t pos = 0; pos < numel; pos++) {
        item_ = PyList_GET_ITEM(items, pos);
        key = PyTuple_GET_ITEM(item_, 0);
        main_ = PyTuple_GET_ITEM(item_, 1);

        for(Py_ssize_t j = 0; j < len; j++) {
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


int parse_apply_args(PyObject *args, PyObject **callable, PyObject **main, PyObject **rest)
{
    PyObject *first = PyTuple_GetSlice(args, 0, 2);
    int parsed = PyArg_ParseTuple(first, "OO|:apply", callable, main);
    Py_DECREF(first);

    if (!parsed)
        return 0;

    if(!PyCallable_Check(*callable)) {
        PyErr_SetString(PyExc_TypeError, "The first argument must be a callable.");
        return 0;
    }

    Py_ssize_t len = PyTuple_GET_SIZE(args);
    *rest = PyTuple_GetSlice(args, 2, len);

    return 1;
}


static PyObject* apply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // from the url at the top: {API 1.2.1} the call mechanism guarantees
    //  to hold a reference to every argument for the duration of the call.
    int safe = 1, star = 1;
    PyObject *callable = NULL, *main = NULL, *rest = NULL;

    //handle `apply(fn, main, *rest, ...)`
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

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
                PyTuple_SET_ITEM(rest_, j, item_);
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
                PyTuple_SET_ITEM(rest_, j, item_);
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
                PyTuple_SET_ITEM(rest_, j, item_);
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


static PyObject* validate(PyObject *self, PyObject *args)
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


static PyMethodDef modapply_methods[] = {
    {
        "apply",
        (PyCFunction) apply,
        METH_VARARGS | METH_KEYWORDS,
        __doc__,
    }, {
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
        "apply",
        NULL,
        -1,
        modapply_methods,
};


PyMODINIT_FUNC
PyInit_apply(void)
{
    return PyModule_Create(&moduledef);
}
