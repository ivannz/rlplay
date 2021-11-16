#include <Python.h>

#include <apply.h>
#include <validate.h>
// https://edcjones.tripod.com/refcount.html
// https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html


static const char *__doc__ = "\n"
    "apply(callable, *objects, _safe=True, _star=True, _finalizer=None, **kwargs)\n"
    "\n"
    "Compute the function using the leaf data of the nested objects as arguments.\n"
    "\n"
    "A `nested object` is either a python object (object, str, numpy array, torch\n"
    "tensor, etc.) or a subclass of one of python's built-in containers (dict,\n"
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
    "\n"
    "    Switching safety off SEGFAULTs if the nested objects do not have\n"
    "    IDENTICAL STRUCTURE, or if `minimality' is violated (see the caveat).\n"
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
    "_finalizer : callable, optional\n"
    "    The finalizer object to be called when a nested container has been\n"
    "    rebuilt. It is NEVER called on the output of `callable`, which is\n"
    "    computed on the leaf python objects. No finalization takes place if\n"
    "    the argument is OMITTED.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "result : a new nested object\n"
    "    The nested object that contains the values returned by `callable`.\n"
    "    Guaranteed to have IDENTICAL structure as the first nested object\n"
    "    in objects.\n"
    "\n"
    "Caveat on `safe=False`\n"
    "----------------------\n"
    "The FIRST object in `*objects` plays a special role: its nested structure\n"
    "determines how all objects are jointly traversed and dictates the structure\n"
    "of the computed result. If safety checks are off, its structure is ALLOWED\n"
    "to be ``minimal'' among the structures of all objects, i.e. lists and tuples\n"
    "of the first object are allowed to be shorter, its dicts' keys may be strict\n"
    "subsets of the corresponding dicts in other objects.\n"
    "\n"
    "    The unsafe procedure SEGFAULTs if this `minimality' is violated,\n"
    "    however safety checks enforce STRICTLY IDENTICAL STRUCTURE.\n"
    "\n"
    "    NOTE: namedtuples are compared as tuples and not as dicts, due to them\n"
    "          being runtime-constructed sub-classes of tuples. Hence for them\n"
    "          only the order matters and not their fields' names.\n"
    "\n"
    "Details\n"
    "-------\n"
    "For a single container `apply` with `_star=True` and omitted `_finalizer`\n"
    "is roughly equivalent to\n"
    "\n"
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


PyObject *PyObject_CallWithSingleArg(
    PyObject *callable,
    PyObject *arg,
    PyObject *kwargs)
{
    // much like `PyObject_CallOneArg`, but with optional kwargs:
    //   create a one-element tuple, then call with it and kwargs
    PyObject *single = PyTuple_New(1);
    if(single == NULL) return NULL;

    // `PyTuple_SET_ITEM` steals reference to `arg`, but we borrowed it from
    //  the caller! Thus we first become its independent owner.
    Py_INCREF(arg);
    PyTuple_SET_ITEM(single, 0, arg);

    // the called object increfs its returned value and transfers the ownership
    //  to the caller, i.e. the act of RETURNING is in itself another reference
    //  see https://docs.python.org/3/extending/extending.html#ownership-rules
    //      https://stackoverflow.com/questions/57661466/
    // (we may check refcounts of `output` and `arg` for `lambda x: x`)
    PyObject *output = PyObject_Call(callable, single, kwargs);

    // No need to relinquish ownership of `arg` (decref), since decrefing
    //  a tuple decrefs all non-NULL items (tuple steals/assumes owenership,
    //  instead of borrowing).
    Py_DECREF(single);

    return output;
}


PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                 bool const safe, bool const star, PyObject *kwargs,
                 PyObject *finalizer);


static PyObject* _apply_dict(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs,
                             PyObject *finalizer)
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
    // Any references returned by `PyDict_Next` are borrowed from the dict
    //     https://docs.python.org/3/c-api/dict.html#c.PyDict_Next
    while (PyDict_Next(main, &pos, &key, &main_)) {
        for(Py_ssize_t j = 0; j < len; j++) {
            // `PyDict_GetItem` and `PyTuple_GET_ITEM` return a borrowed reference
            //     https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItem
            item_ = PyDict_GetItem(PyTuple_GET_ITEM(rest, j), key);

            // `PyTuple_SetItem` decrefs any non-NULL item already in the tuple
            //  at the affected position. In contrast, `PyTuple_SET_ITEM` does
            //  NOT discard references to items being replaced!
            //    https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
            Py_XDECREF(PyTuple_GET_ITEM(rest_, j));

            // a tuple assumes ownership of, or 'steals', the reference, owned
            //  by a dict from `rest`, so we incref it for protection.
            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        // `result` is a new object, for which we are now responsible
        result = _apply(callable, main_, rest_, safe, star, kwargs, finalizer);
        if(result == NULL) {
            Py_DECREF(rest_);

            // decrefing a dict also aplies decref to its contents
            Py_DECREF(output);
            return NULL;
        }

        // dict's setitem DOES NOT steal references to `val` and, apparently,
        //  to `key`, i.e. does an incref of its own (both value and the key),
        //  which is why `_apply_dict` logic is different from `_tuple`.
        //     https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem
        PyDict_SetItem(output, key, result);

        // decref the result, so that only `output` owns a ref
        Py_DECREF(result);
    }

    // decrefing a tuple also decrefs all its items
    Py_DECREF(rest_);

    return output;
}


static PyObject* _apply_tuple(PyObject *callable, PyObject *main, PyObject *rest,
                              bool const safe, bool const star, PyObject *kwargs,
                              PyObject *finalizer)
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
            // `PyTuple_GET_ITEM` returns a borrowed reference (from the tuple)
            //     https://docs.python.org/3/c-api/tuple.html#c.PyTuple_GET_ITEM
            item_ = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

            Py_XDECREF(PyTuple_GET_ITEM(rest_, j));

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs, finalizer);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        // `PyTuple_SET_ITEM` steals references and does NOT discard refs
        // of displaced objects.
        //     https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
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

    // since `namedtuple`-s are immutable and derived from `tuple`,
    //  we can just call `tp_new` on them
    // XXX fix this if the namedtuple's implementation changes
    // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_new
    PyObject *namedtuple = Py_TYPE(main)->tp_new(Py_TYPE(main), output, NULL);
    Py_DECREF(output);

    return namedtuple;
}


static PyObject* _apply_list(PyObject *callable, PyObject *main, PyObject *rest,
                             bool const safe, bool const star, PyObject *kwargs,
                             PyObject *finalizer)
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
            // `PyList_GET_ITEM` returns a borrowed reference (from the list)
            //     https://docs.python.org/3/c-api/list.html#c.PyList_GET_ITEM
            item_ = PyList_GET_ITEM(PyTuple_GET_ITEM(rest, j), pos);

            Py_XDECREF(PyTuple_GET_ITEM(rest_, j));

            Py_INCREF(item_);
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        result = _apply(callable, main_, rest_, safe, star, kwargs, finalizer);
        if(result == NULL) {
            Py_DECREF(rest_);
            Py_DECREF(output);
            return NULL;
        }

        // Like `PyList_SetItem`, `PyList_SET_ITEM` steals the reference from
        // us. However, unlike it `_SET_ITEM` DOES NOT discard refs of
        // displaced objects. We're ok, because `output` is a NEW list.
        //     https://docs.python.org/3/c-api/list.html#c.PyList_SET_ITEM
        PyList_SET_ITEM(output, pos, result);
    }

    Py_DECREF(rest_);

    return output;
}


static PyObject* _apply_mapping(PyObject *callable, PyObject *main, PyObject *rest,
                                bool const safe, bool const star, PyObject *kwargs,
                                PyObject *finalizer)
{
    // XXX it's unlikely that we will ever use this branch, because as docs say
    //  it is impossible to know the type of keys of a mapping at runtime, hence
    //  lists, tuples, dicts and any objects with `__getitem__` are mappings
    //  according to `PyMapping_Check`.
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
            // `PyObject_GetItem` yields a new reference
            //     https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem
            item_ = PyObject_GetItem(PyTuple_GET_ITEM(rest, j), key);

            Py_XDECREF(PyTuple_GET_ITEM(rest_, j));
            PyTuple_SET_ITEM(rest_, j, item_);
        }

        Py_DECREF(result);

        result = _apply(callable, main_, rest_, safe, star, kwargs, finalizer);
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
    } else {
        output = PyObject_CallWithSingleArg(callable, args, kwargs);
    }
    Py_DECREF(args);

    return output;
}


PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                 bool const safe, bool const star, PyObject *kwargs,
                 PyObject *finalizer)
{
    PyObject *result;

    if(PyDict_Check(main)) {
        if(safe)
            if(!_validate_dict(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_dict(callable, main, rest, safe, star, kwargs, finalizer);
        Py_LeaveRecursiveCall();

    } else if(PyTuple_Check(main)) {
        if(safe)
            if(!_validate_tuple(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_tuple(callable, main, rest, safe, star, kwargs, finalizer);
        Py_LeaveRecursiveCall();

    } else if(PyList_Check(main)) {
        if(safe)
            if(!_validate_list(main, rest))
                return NULL;

        if(Py_EnterRecursiveCall("")) return NULL;
        result = _apply_list(callable, main, rest, safe, star, kwargs, finalizer);
        Py_LeaveRecursiveCall();

    } else {
        // The base case, i.e. having reached the leaf objects (non containers)
        // is non recursive
        return _apply_base(callable, main, rest, star, kwargs);
    }

    // bypass the finalizer if _apply_* failed and bubble up the exception
    if(finalizer == NULL || result == NULL)
        return result;

    // The finalizer is only called on the inner/nested containers, and never
    //  on the leaf data
    PyObject *output = PyObject_CallWithSingleArg(finalizer, result, NULL);
    Py_DECREF(result);

    return output;
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


PyObject* apply(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // from the url at the top: {API 1.2.1} the call mechanism guarantees
    //  to hold a reference to every argument for the duration of the call.
    int safe = 1, star = 1;
    PyObject *callable = NULL, *main = NULL, *rest = NULL, *finalizer=NULL;

    //handle `apply(fn, main, *rest, ...)`
    if(!parse_apply_args(args, &callable, &main, &rest))
        return NULL;

    //handle `apply(..., *, _star, _safe, **kwargs)`
    if (kwargs) {
        static char *kwlist[] = {"_safe", "_star", "_finalizer", NULL};

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

            // PyDict_SetItem uses `Py_INCREF() to become an independent owner`
            //  see https://docs.python.org/3/extending/extending.html#ownership-rules
            PyDict_SetItemString(own, kwlist[p], arg);
            PyDict_DelItemString(kwargs, kwlist[p]);
        }

        // PyArg_ParseTupleAndKeywords does not do anythin with the owenrship
        //  of `PyObject`, https://docs.python.org/3/c-api/arg.html#other-objects
        // Thus we hold on to the `finalizer` in case its only ref was
        //  the `kwargs`, which we tinkered with just above.
        int parsed = PyArg_ParseTupleAndKeywords(
                empty, own, "|$ppO:apply", kwlist, &safe, &star, &finalizer);
        Py_XINCREF(finalizer);

        Py_DECREF(empty);
        Py_DECREF(own);
        if (!parsed) return NULL;

        if(finalizer != NULL && !PyCallable_Check(finalizer)) {
            PyErr_SetString(PyExc_TypeError, "The finalizer must be a callable.");
            return NULL;
        }
    }

    PyObject *result = _apply(callable, main, rest, safe, star, kwargs, finalizer);
    Py_DECREF(rest);
    Py_XDECREF(finalizer);

    return result;
}


const PyMethodDef def_apply = {
    "apply",
    (PyCFunction) apply,
    METH_VARARGS | METH_KEYWORDS,
    __doc__,
};
