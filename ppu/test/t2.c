#include <stdio.h>
extern int hgDeviceCount(void);
extern int hgSetDevice(int);
extern void* hgMemAlloc(unsigned long);
extern int hgMemFree(void*);
extern int hgMemcpyH2D(void*, const void*, unsigned long, void*);
extern int hgMemcpyD2H(void*, const void*, unsigned long, void*);
extern int hgDeviceSynchronize(void);
extern void* hgStreamCreate(void);
extern int hgStreamSynchronize(void*);
extern int hgStreamDestroy(void*);
int main(){
    int n = hgDeviceCount();
    printf("devices=%d\n", n);
    printf("setdev=%d\n", hgSetDevice(0));
    void* p = hgMemAlloc(1<<20);
    printf("alloc=%p\n", p);
    if(p){
        float h[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float out[4] = {0,0,0,0};
        printf("h2d=%d\n", hgMemcpyH2D(p, h, 16, 0));
        printf("d2h=%d\n", hgMemcpyD2H(out, p, 16, 0));
        printf("data=%.1f,%.1f,%.1f,%.1f\n", out[0], out[1], out[2], out[3]);
        printf("free=%d\n", hgMemFree(p));
    }
    void* s = hgStreamCreate();
    printf("stream=%p\n", s);
    if(s){ printf("ssync=%d\n", hgStreamSynchronize(s)); printf("sdest=%d\n", hgStreamDestroy(s)); }
    printf("sync=%d\n", hgDeviceSynchronize());
    return 0;
}
