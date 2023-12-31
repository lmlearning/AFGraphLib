#!/bin/bash
# (c)2014 Federico Cerutti <federico.cerutti@acm.org> --- MIT LICENCE
# adapted 2016 by Thomas Linsbichler <linsbich@dbai.tuwien.ac.at> --- MIT LICENSE
# adapted 2018 by Francesco Santini <francesco.santini@dmi.unipg.it> --- MIT LICENSE
# adapted 2018 by Theofrastos Mantadelis <theo.mantadelis@dmi.unipg.it> --- MIT LICENSE
# Generic script interface for ICCMA 2019
# Please feel free to customize it for your own solver
# In the 2019 adaptation we use "sh" instead of "bash" because Alpine Linux does not natively support it


# function for echoing on standard error
function echoerr()
{
    # to remove standard error echoing, please comment the following line
    echo "$@" 1>&2; 
}

################################################################
# C O N F I G U R A T I O N
# 
# this script must be customized by defining:
# 1) procedure for printing author and version information of the solver
#    (function "information")
# 2) suitable procedures for invoking your solver (function "solver");
# 3) suitable procedures for parsing your solver's output 
#    (function "parse_output");
# 4) list of supported format (array "formats");
# 5) list of supported problems (array "problems").

# output information
function information()
{
    # to be adapted
    echo "AFGCN 0.2"
    echo "Lars Malmqvist <lm1775@york.ac.uk>"
}

# how to invoke your solver: this function must be customized
# these variables are taken from the environment, and they are set in Dockerfile
# that has been packaged in the container
function solver
{

    fileinput=$1    # input file with correct path

    #format=$2    # format of the input file (see below)

    task=$2        # task to solve (see below)

    additional=$3    # additional information, i.e. name of an argument

    DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd) 
    SOLVER="solver.py"
    python -W ignore::UserWarning: "$DIR"/"$SOLVER"  --filepath $fileinput --task $task --argument $additional 
}


# How to invoke your dynamic solver: this function must be customized
# these variables are taken from the environment, and they are set in Dockerfile
# that has been packaged in the container
function dynamic_solver
{

    fileinput=$1    # input file with correct path

    filemod=$2    # modificarions file with correct path

    format=$3    # format of the input file (see below)

    task=$4        # task to solve (see below)

    additional=$5    # additional information, i.e. name of an argument

    DIR="$PWD/"
    SOLVER="conarg"


    # Example for ConArg that doesn't support dynamic
    # When solver doesn't support dynamic you can use the autogenerated files
    # with .0 ... .n extensions as input files
    echo $task
    for f in $fileinput.*
    do
    case $task in
            "DC-CO-D")
                $DIR$SOLVER -q -p -e complete -c $additional $f # $fileinput -m $filemod
            ;;
            "DC-PR-D")
                $DIR$SOLVER -q -p -e preferred -c $additional $f # $fileinput -m $filemod
            ;;
            "DC-ST-D")
                $DIR$SOLVER -q -p -e stable -c $additional $f # $fileinput -m $filemod
            ;;
            "DC-GR-D")
                $DIR$SOLVER -q -p -e grounded -c $additional $f # $fileinput -m $filemod
            ;;
            "DS-CO-D")
                $DIR$SOLVER -q -p -e complete -s $additional $f # $fileinput -m $filemod
            ;;
            "DS-PR-D")
                $DIR$SOLVER -q -p -e preferred -s $additional $f # $fileinput -m $filemod
            ;;
            "DS-ST-D")
                $DIR$SOLVER -q -p -e stable -s $additional $f # $fileinput -m $filemod
            ;;
            "EE-CO-D")
                $DIR$SOLVER -q -p -e complete $f # $fileinput -m $filemod
            ;;
            "EE-PR-D")
                $DIR$SOLVER -q -p -e preferred $f # $fileinput -m $filemod
            ;;
            "EE-ST-D")
                $DIR$SOLVER -q -p -e stable $f # $fileinput -m $filemod
            ;;
            "SE-CO-D")
                $DIR$SOLVER -q -p -e complete -n 1 $f # $fileinput -m $filemod
            ;;
            "SE-PR-D")
                $DIR$SOLVER -q -p -e preferred -n 1 $f # $fileinput -m $filemod
            ;;
            "SE-ST-D")
                $DIR$SOLVER -q -p -e stable -n 1 $f # $fileinput -m $filemod
            ;;
            "SE-GR-D")
                $DIR$SOLVER -q -p -e grounded -n 1 $f # $fileinput -m $filemod
            ;;
            *)
                echo "Task $task not supported"

            ;;
        esac
    done
    
}


# accepted formats: please comment those unsupported
formats=""
formats="${formats} apx" # "aspartix" format
formats="${formats} tgf" # trivial graph format

# task that are supported: please comment those unsupported

#+------------------------------------------------------------------------+
#|         I C C M A   2 0 1 9   L I S T   O F   P R O B L E M S          |
#|                                                                        |
tasks=""
tasks="${tasks} DC-CO"     # Decide credulously according to Complete semantics
tasks="${tasks} DS-CO"     # Decide skeptically according to Complete semantics
#tasks="${tasks} SE-CO"     # Enumerate some extension according to Complete semantics
#tasks="${tasks} EE-CO"     # Enumerate all the extensions according to Complete semantics
tasks="${tasks} DC-PR"     # Decide credulously according to Preferred semantics
tasks="${tasks} DS-PR"     # Decide skeptically according to Preferred semantics
#tasks="${tasks} SE-PR"     # Enumerate some extension according to Preferred semantics
#tasks="${tasks} EE-PR"     # Enumerate all the extensions according to Preferred semantics
tasks="${tasks} DC-ST"     # Decide credulously according to Stable semantics
tasks="${tasks} DS-ST"     # Decide skeptically according to Stable semantics
#tasks="${tasks} SE-ST"     # Enumerate some extension according to Stable semantics
#tasks="${tasks} EE-ST"     # Enumerate all the extensions according to Stable semantics
tasks="${tasks} DC-SST"     # Decide credulously according to Semi-stable semantics
tasks="${tasks} DS-SST"     # Decide skeptically according to Semi-stable semantics
#tasks="${tasks} SE-SST"     # Enumerate some extension according to Semi-stable semantics
#tasks="${tasks} EE-SST"     # Enumerate all the extensions according to Semi-stable semantics
tasks="${tasks} DC-STG"     # Decide credulously according to Stage semantics
tasks="${tasks} DS-STG"     # Decide skeptically according to Stage semantics
#tasks="${tasks} EE-STG"     # Enumerate all the extensions according to Stage semantics
#tasks="${tasks} SE-STG"     # Enumerate some extension according to Stage semantics
#tasks="${tasks} DC-GR"     # Decide credulously according to Grounded semantics
#tasks="${tasks} SE-GR"     # Enumerate some extension according to Grounded semantics
tasks="${tasks} DC-ID"     # Decide credulously according to Ideal semantics
tasks="${tasks} DS-ID"     # Decide credulously according to Ideal semantics
#tasks="${tasks} SE-ID"     # Enumerate some extension according to Ideal semantics
#tasks="${tasks} DC-CO-D"     # -Dynamic- Decide credulously according to Complete semantics
#tasks="${tasks} DS-CO-D"     # -Dynamic- Decide skeptically according to Complete semantics
#tasks="${tasks} SE-CO-D"     # -Dynamic- Enumerate some extension according to Complete semantics
#tasks="${tasks} EE-CO-D"     # -Dynamic- Enumerate all the extensions according to Complete semantics
#tasks="${tasks} DC-PR-D"     # -Dynamic- Decide credulously according to Preferred semantics
#tasks="${tasks} DS-PR-D"     # -Dynamic- Decide skeptically according to Preferred semantics
#tasks="${tasks} SE-PR-D"     # -Dynamic- Enumerate some extension according to Preferred semantics
#tasks="${tasks} EE-PR-D"     # -Dynamic- Enumerate all the extensions according to Preferred semantics
#tasks="${tasks} DC-ST-D"     # -Dynamic- Decide credulously according to Stable semantics
#tasks="${tasks} DS-ST-D"     # -Dynamic- Decide skeptically according to Stable semantics
#tasks="${tasks} SE-ST-D"     # -Dynamic- Enumerate some extension according to Stable semantics
#tasks="${tasks} EE-ST-D"     # -Dynamic- Enumerate all the extensions according to Stable semantics
#tasks="${tasks} DC-GR-D"     # -Dynamic- Decide credulously according to Grounded semantics
#tasks="${tasks} SE-GR-D"     # -Dynamic- Enumerate some extension according to Grounded semantics
#|                                                                        |
#|  E N D   O F   I C C M A   2 0 1 9   L I S T   O F   P R O B L E M S   |
#+------------------------------------------------------------------------+


function list_output
{
    check_something_printed=false
    printf "["
    if [[ "$1" = "1" ]];
        then
        for format in ${formats}; do
            if [ "$check_something_printed" = true ];
            then
                printf ", "
            fi
            printf "%s" $format
            check_something_printed=true
        done
        printf "]\n"
    elif [[ "$1" = "2" ]];
        then
        for task in ${tasks}; do
            if [ "$check_something_printed" = true ];
            then
                printf ", "
            fi
            printf "%s" $task
            check_something_printed=true
        done
        printf "]\n"
    fi
}



# how to parse the output of your solver in order to be compliant with ICCMA 2019:
# this function must be customized
# solutions must be of the following form:
#    [arg1,arg2,...,argN]                     for single extension (SE)
#    [[arg1,arg2,...,argN],[...],...]         for extension(s) enumeration (EE)
#    YES/NO                                   for decision problems (DC and DS)
#       [[arg1,...argN]],[[...],...],[[...],...] for Dung's triatholon (D3)
function parse_output()
{
    task=$1
    output="$2"

    echoerr "original output: $output"

    #example of parsing EE-tasks for MySolver0.8.15, which returns "{arg1,arg2,...}\n{...}\n..."
    if [[ "$task" == "EE-"* ]];
    then
        printf "["
        echo $output | sed 's/{/[/g' | sed 's/}/]/g' | tr -d '\n' \
               | sed 's/\]\[/\],\[/g' | sed 's/\] \[/\],\[/g'
        printf "]"
    elif [ "$task" = "D3" ];
    then
        echo $output
    #
    # other tasks
    #
    else
        echoerr "unsupported format or task"
        exit 1
    fi
}




function main
{

    if [ "$#" = "0" ]
    then
        information
        exit 0
    fi

    local local_problem=""
    local local_fileinput=""
    local local_format=""
    local local_additional=""
    local local_filemod=""
    local local_task=""
    local local_task_valid=""

    while [ "$1" != "" ]; do
        case $1 in
          "--formats")
        list_output 1
        exit 0
        ;;
          "--problems")
        list_output 2
        exit 0
        ;;
          "-p")
        shift
        local_problem=$1
        ;;
          "-f")
        shift
        local_fileinput=$1
        ;;
          "-fo")
        shift
        local_format=$1
        ;;
          "-a")
        shift
        local_additional=$1
        ;;
          "-m")
        shift
        local_filemod=$1
        ;;
        esac
        shift
    done

    if [ -z $local_problem ]
    then
        echo "Task missing"
        exit 0
    else
        for local_task in ${tasks}; do
            if [ $local_task = $local_problem ]
            then
              local_task_valid="true"
            fi
        done
        if [ -z $local_task_valid ]
        then
            echo "Invalid task"
            exit 0
        fi
    fi

    if [ -z $local_fileinput ]
    then
        echo "Input file missing"
        exit 0
    fi

    if [ -z $local_filemod ]
    then
    
        res=$(solver $local_fileinput $local_format $local_problem $local_additional)
    
    else
        
        res=$(dynamic_solver $local_fileinput $local_filemod $local_format $local_problem $local_additional)

    fi
  
    echo "$res"

    # If your solver output is not natively compliant with ICCMA 2019, use this function to adapt it
    #parse_output $local_problem "$res"

}

main $@
exit 0
