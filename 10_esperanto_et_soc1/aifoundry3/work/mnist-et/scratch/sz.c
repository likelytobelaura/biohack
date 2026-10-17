#include <stdio.h>
#include "mnist_args.h"
#define K(a,b) printf("%-6s %7.1f KB\n", #a, ((b)-(a))/1024.0)
int main(){K(MN_S_W,MN_S_T);K(MN_S_T,MN_S_V);K(MN_S_V,MN_S_GP);K(MN_S_GP,MN_S_GS);K(MN_S_GS,MN_S_A0);K(MN_S_A0,MN_S_Z);K(MN_S_Z,MN_S_LP);K(MN_S_LP,MN_S_PRIV);K(MN_S_PRIV,MN_S_STAGE);K(MN_S_STAGE,MN_S_END);printf("start %llu KB end %llu KB\n",MN_S_W/1024,MN_S_END/1024);}
