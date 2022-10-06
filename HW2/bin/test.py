import ctypes

clib_path = "vf3.so"

clib = ctypes.CDLL(clib_path)
clib.test.restype = ctypes.c_int
subgraph_txt = open(
    "test/bvg1.sub.grf",
    "rb",
).read()
graph_txt = open(
    "test/bvg1.sub.grf.1",
    "rb",
).read()
# a = ctypes.c_char_p(b"hello")
# b = ctypes.c_char_p(b"world")
a = ctypes.c_char_p(subgraph_txt)
b = ctypes.c_char_p(graph_txt)
print("Changed edge label ->")
print("[a,b]", clib.test(a, b))
print("Same edge label ->")

print("[a,a]", clib.test(a, a))
print("[b,b]", clib.test(b, b))
