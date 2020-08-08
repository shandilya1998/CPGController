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
## Experiment 2
In Experiment 1, the signals were not normalized. 
- N = 500
- Tst = 60
- Tsw = 20
- dt = 0.001
- lr = 1e-3
- cyclic learning rate
- num\_osc=20
- num\_h=50
- num\_out=8
- fundamental frequency is computed using fft 
The followin plot is the error plot for training
![Error Plot](../images/training_plot_output_mlp_exp2.png)
The following plot is the comparison of the reconstructed signal and the original signal
![Signal Reconstruction Plot](../images/pred_vs_ideal_exp2.png)
## Experiment 3
- N = 500
- Tst = 60
- Tsw = 20
- dt = 0.001
- lr = 1e-3
- cyclic learning rate
- num\_osc=40
- num\_h=200
- num\_out=8
- nepochs = 10000
- fundamental frequency is computed using fft
The followin plot is the error plot for training
![Error Plot](../images/training_plot_output_mlp_exp3.png)
The following plot is the comparison of the reconstructed signal and the original signal
![Signal Reconstruction Plot](../images/pred_vs_ideal_exp3.png)
## Experiment 3
- N = 500 
- Tst = 60
- Tsw = 20
- dt = 0.001
- lr = 1e-3
- cyclic learning rate
- num\_osc=20
- num\_h=200
- num\_out=8
- nepochs = 30000
- fundamental frequency is computed using fft 
The followin plot is the error plot for training
![Error Plot](../images/training_plot_output_mlp_exp4.png)
The following plot is the comparison of the reconstructed signal and the original signal
![Signal Reconstruction Plot](../images/pred_vs_ideal_exp4.png)

