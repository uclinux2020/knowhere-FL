#include <dlfcn.h>
#include <stdio.h>

typedef int (*i_p)(int);
typedef int (*sc_p)(void**, unsigned);

static void probe(const char* path, int flags, const char* tag) {
    void* h = dlopen(path, flags);
    printf("[%s] dlopen(%s)=%p err=%s\n", tag, path, h, h ? "(null)" : dlerror());
    if (!h) return;
    i_p p_init = (i_p)dlsym(h, "hgInit");
    i_p p_set  = (i_p)dlsym(h, "hgSetDevice");
    sc_p p_sc  = (sc_p)dlsym(h, "hgStreamCreate");
    printf("[%s] init=%p set=%p sc=%p\n", tag, p_init, p_set, p_sc);
    if (p_init) printf("[%s] hgInit(0)=%d\n", tag, p_init(0));
    if (p_set)  printf("[%s] hgSetDevice(0)=%d\n", tag, p_set(0));
    if (p_sc) {
        void* s = 0;
        int r = p_sc(&s, 0);
        printf("[%s] hgStreamCreate=%d s=%p\n", tag, r, s);
    }
}

int main(void) {
    const char* W = "/usr/local/PPU_SDK/lib/libhg_wrapper.so";
    probe(W, RTLD_NOW | RTLD_LOCAL, "plain");
#ifdef RTLD_DEEPBIND
    probe(W, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND, "deepbind");
#endif
    printf("T5_DONE\n");
    return 0;
}
