#include <stdio.h>
#include "mnist_args.h"
int main(){printf("end %llu KB of %llu; W %llx T %llx V %llx GP %llx GS %llx A0 %llx Z %llx PRIV %llx\n",(unsigned long long)MN_S_END/1024,(unsigned long long)MN_SCP_BYTES/1024,MN_S_W,MN_S_T,MN_S_V,MN_S_GP,MN_S_GS,MN_S_A0,MN_S_Z,MN_S_PRIV);}
