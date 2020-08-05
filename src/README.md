# OutputMLP training for multi speed gait generation
Experiments involve training the OutputMLP taking as input, fourier components obtained by using the integer multiples of fundamental frequency of the output signal. 
## Experiment 1
Training the OutputMLP to reconstruct the control signal for a single speed.
The following assumptions hold true for this experiment:->
- The motion will be with a single speed
- No deviation from a straight line
- The aforementioned two points will lead to same fundamental frequency of the control signals for all legs
