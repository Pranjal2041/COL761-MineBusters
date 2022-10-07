#!/bin/bash
set -e;
data=$(realpath ${1:-../data/Yeast/167.txt_graph})
num_graphs=$(cat $data | grep -c '#')
log_file=$(mktemp)
time_file="out_times.csv"
plot_file=$(realpath ${2:-out_times.png})
_python="python3"

echo "Num graphs: $num_graphs"
echo "Log file: " $log_file

function make_input(){
  program=$1
  input_file=$(dirname $data)/$program.txt
  if [[ -f $input_file ]]; then
    return
  fi
  local python_cmd="$_python q1.py -i $data -o $input_file --to $program"
  # echo $python_cmd
  eval $python_cmd
}

for program in gspan fsg gaston; do
  make_input $program

  for support in 0.05 0.10 0.25 0.50 0.95; do

    if [[ $program = "fsg" ]]; then
      support=$(echo $support*100 | bc)
      cmd="./bin/fsg -s $support $input_file"
    elif [[ $program = "gaston" ]]; then
      support=$(printf %.0f $(echo $support*$num_graphs | bc))
      cmd="./bin/gaston $support $input_file"
    elif [[ $program = "gspan" ]]; then
      support=$support
      cmd="./bin/gspan -f $input_file -s $support"
    fi;
    echo will execute command: $cmd
    _t=`{ time $cmd >> $log_file; } 2>&1 | awk '/real/{print $2}'`
    echo "Execution time: $_t" 
    echo "$program,$support,$_t" >> $time_file
  done;
done;

echo All logs written to: $log_file
echo Times calculated and written to $time_file

$_python q1.py --plot-input $time_file --plot-output $plot_file

echo Plot output to $plot_file
exit 0;