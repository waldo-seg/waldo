#!/bin/bash
# Copyright   2018 Ashish Arora

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi

  shift
done

nj=4
cmd=run.pl

echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./scripts/parse_options.sh || exit 1;

data=$save_dir
log_dir=$data/log

mkdir -p $log_dir

for n in $(seq $nj); do
  mkdir -p $data/$n
done

for dir in images labels; do
  ls $( eval "echo \$${dir}" ) > ${log_dir}/${dir}_fn.txt
  
  total_lines=$(wc -l <${log_dir}/${dir}_fn.txt)
  ((lines_per_file = (total_lines + nj - 1) / nj))
  split --lines=${lines_per_file} -a 1 --numeric-suffixes=1 ${log_dir}/${dir}_fn.txt ${log_dir}/${dir}_fn_

  rm ${log_dir}/${dir}_fn.txt
done

$cmd JOB=1:$nj $log_dir/extract_masks.JOB.log \
  local/get_masks_from_page_image.py $images $labels $data/JOB $log_dir/lines.JOB.scp \
  --images_fn=$log_dir/images_fn_JOB --labels_fn=$log_dir/labels_fn_JOB \
  || exit 1;

## concatenate the .scp files together.
for n in $(seq $nj); do
  cat $data/$n/images.scp || exit 1;
done > $data/images.scp || exit 1

for n in $(seq $nj); do
  rm $log_dir/images_fn_$n
  rm $log_dir/labels_fn_$n
done
