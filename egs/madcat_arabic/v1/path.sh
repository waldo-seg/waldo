export PATH="${HOME}/miniconda3/bin:$PATH"
export PYTHONPATH="${PYTHONPATH}:scripts"
export PATH=$PWD/../../../scripts/parallel/:$PATH
export PATH=$PWD/../../../scripts/waldo:$PATH
export PATH=$PWD/../../../scripts:$PATH
export PYTHONPATH=${PYTHONPATH}:$PWD/../../../scripts

# Here is a shell snippet that removes duplicates from $PATH. 
# Ref: http://linuxg.net/oneliners-for-removing-the-duplicates-in-your-path/
export PATH=`echo -n $PATH | awk -v RS=: -v ORS=: '{ if (!arr[$0]++) { print $0 } }'`
