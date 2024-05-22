# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF='/home/qinjinghui/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
  DEFAULT_CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
  DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
fi

CANN_INSTALL_PATH=${1:-${DEFAULT_CANN_INSTALL_PATH}}

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 导入依赖库
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/openblas/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64_64-linux-gnu
# export LD_PRELOAD=$LD_PRELOAD:/usr/local/python3.9.2/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息