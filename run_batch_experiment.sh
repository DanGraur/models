#!/usr/bin/env bash
# Prepare parameters
model=${1:-"dlrm"}
batch_sizes="$( for i in {4..14}; do echo $((2**$i)); done)"

# Prepare target experiment documents
EXPERIMENT_FOLDER="experiments/tensorflow_$(date +%F_%H_%M_%S_%3N)"
RESULT_FILE="${EXPERIMENT_FOLDER}/results.csv"
mkdir -p ${EXPERIMENT_FOLDER}
COMMAND="run.sh"
echo "system,sample_count,batch,throughput" > ${RESULT_FILE}

# Run experiment
for i in ${batch_sizes}; do
  echo "Starting experiment for ${i} batch size..."
  experiment_log="${EXPERIMENT_FOLDER}/batch_${i}.log"
  # experiment_stderr_log="${EXPERIMENT_FOLDER}/batch_${i}.stderr.log"
  . run.sh ${model} ${i}  > >(tee ${experiment_log}) 2> /dev/null
  throughput=$( cat ${experiment_log} | tail -n 1 | awk '{print $16}' )
  echo "tensorflow,10000000,${i},${throughput}" >> ${RESULT_FILE}
  echo "Finished experiment for ${i} batch size!\n"
done

echo "Results now available in: ${RESULT_FILE}"
