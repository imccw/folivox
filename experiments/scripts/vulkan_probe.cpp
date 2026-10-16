// Small direct Vulkan graphs checked against a scalar CPU reference.
// No scheduler or CPU fallback: every compute node must support Vulkan.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void set(ggml_tensor *t, const std::vector<float> &v) {
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        for (size_t i=0; i<v.size(); ++i) h[i]=ggml_fp32_to_fp16(v[i]);
        ggml_backend_tensor_set(t,h.data(),0,h.size()*sizeof(h[0]));
    } else ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));
}

static void check(ggml_backend_t backend, bool half, bool conv) {
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto type=half?GGML_TYPE_F16:GGML_TYPE_F32;
    ggml_tensor *a,*b,*out;
    std::vector<float> av,bv,ref;
    if (!conv) {
        constexpr int k=32,m=8,n=8;
        a=ggml_new_tensor_2d(ctx,type,k,m);
        b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        av.resize(k*m); bv.resize(k*n); ref.resize(m*n);
        for (size_t i=0;i<av.size();++i) av[i]=float(int(i%9)-4)/8;
        for (size_t i=0;i<bv.size();++i) bv[i]=float(int(i%7)-3)/8;
        for (int j=0;j<n;++j) for (int i=0;i<m;++i) for(int l=0;l<k;++l)
            ref[j*m+i]+=av[i*k+l]*bv[j*k+l];
        out=ggml_mul_mat(ctx,a,b);
    } else {
        // Two input channels, four output channels, length 16, three taps.
        constexpr int k=3,ic=2,oc=4,len=16;
        a=ggml_new_tensor_3d(ctx,type,k,ic,oc);
        b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,len,ic);
        av.resize(k*ic*oc);bv.resize(len*ic);ref.resize(len*oc);
        for(size_t i=0;i<av.size();++i) av[i]=float(int(i%9)-4)/8;
        for(size_t i=0;i<bv.size();++i) bv[i]=float(int(i%7)-3)/8;
        for(int o=0;o<oc;++o) for(int t=0;t<len;++t) for(int c=0;c<ic;++c) for(int l=0;l<k;++l) {
            int x=t+l-1;
            if(x>=0 && x<len) ref[o*len+t]+=av[(o*ic+c)*k+l]*bv[c*len+x];
        }
        out=ggml_conv_1d(ctx,a,b,1,1,1);
    }
    auto graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if(!ggml_backend_supports_op(backend,node))
            throw std::runtime_error(std::string("Unsupported direct Vulkan op: ")+ggml_op_name(node->op));
    }
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer) throw std::runtime_error("GPU allocation failed");
    set(a,av);set(b,bv);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("Vulkan graph failed");
    ggml_backend_synchronize(backend);
    std::vector<float> result(ref.size());
    ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
    float error=0;
    for(size_t i=0;i<ref.size();++i) {
        if(!std::isfinite(result[i])) throw std::runtime_error("nonfinite GPU result");
        error=std::max(error,std::abs(ref[i]-result[i]));
    }
    printf("%s %s max_abs_error=%.8f %s\n",conv?"conv1d":"matmul",half?"f16":"f32",error,error<0.001?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
    if(error>=0.001) throw std::runtime_error("GPU result differs from scalar reference");
}
int main() {
    try {
        auto backend=ggml_backend_vk_init(0);
        if(!backend) throw std::runtime_error("Vulkan initialization failed");
        auto dev=ggml_backend_get_device(backend);
        auto desc=ggml_backend_dev_description(dev);
        printf("device=%s backend=%s\n",desc,ggml_backend_name(backend));
        if(!strstr(desc,"780M") || ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU)
            throw std::runtime_error("Expected hardware Radeon 780M");
        check(backend,false,false); check(backend,true,false);
        check(backend,false,true); check(backend,true,true);
        ggml_backend_free(backend);
        puts("All direct Vulkan computations passed; no CPU fallback.");
    } catch(const std::exception &e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
