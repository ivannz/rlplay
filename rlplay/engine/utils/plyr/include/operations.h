PyObject* getitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* setitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* xgetitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* xsetitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* is_sequence(PyObject *self, PyObject *object);
PyObject* is_mapping(PyObject *self, PyObject *object);

PyObject* dict_getrefs(PyObject *self, PyObject *dict);
PyObject* dict_clone(PyObject *self, PyObject *dict);

extern const PyMethodDef def_getitem;
extern const PyMethodDef def_setitem;

extern const PyMethodDef def_xgetitem;
extern const PyMethodDef def_xsetitem;

extern const PyMethodDef def_is_sequence;
extern const PyMethodDef def_is_mapping;

extern const PyMethodDef def_dict_getrefs;
extern const PyMethodDef def_dict_clone;
