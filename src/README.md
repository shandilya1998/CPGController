# OutputMLP training for multi speed gait generation
Experiments involve training the OutputMLP taking as input, fourier components obtained by using the integer multiples of fundamental frequency of the output signal. 
## Experiment 1
Training the OutputMLP to reconstruct the control signal for a single speed.
The following assumptions hold true for this experiment:->
- The motion will be with a single speed
- No deviation from a straight line
- The aforementioned two points will lead to same fundamental frequency of the control signals for all legs
The learning rate for this experiment was 1e-3. 
It was observed from the loss plot that loss after a sudded decrease for the first time does not fall greatly. 
A larger learning rate is required
![Error Plot](../images/training_plot_output_mlp_exp1.png)
![Signal Reconstruction Plot](../images/training_plot_output_mlp_exp1.png)
## Experiment 2
N = 500
Tst = 60
Tsw = 20
dt = 0.001
lr = 1e-3
cyclic learning rate
![Error Plot](../images/training_plot_output_mlp_exp2.png)
![Signal Reconstruction Plot](../images/training_plot_output_mlp_exp2.png)
