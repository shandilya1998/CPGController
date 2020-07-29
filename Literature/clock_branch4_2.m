clear all
close all

T = 2*pi/3.5;
dt = 0.001;
N = floor(T/dt);
t = (0:N-1)*dt;
nepoch = 50000;
eparr = 1:nepoch;

%%%% Network architecture %%%%
nos = 3; % no of oscillators
J = 5;   % no of hidden layer neurons in the I/P MLP
I = 3;   % no of I/P nodes

%%%% desired I/P-O/P %%%%
X = rand(I,1);
O = zeros(1,N);
%omegaip = 2*pi*rand(nos,1);  % frequency components in the I/P signal
omegaip = [2; 3.5; 1];
phiip = [pi/3; pi/4; pi/7];
A = [0.2; 0.8; 0.5];
for i =1:nos
    O = O + A(i)*cos(omegaip(i)*t + repmat(phiip(i),[1,N]));
end
Y = zeros(1,N);
r = zeros(nos,N);
phi = zeros(nos,N);
r0 = ones(nos,1);
phi0 = zeros(nos,1);
omegaarr = zeros(nos,nepoch);
err = zeros(1,nepoch);

%%%% activation function parameter %%%%
% parameter of I/P MLP activation function
relum = 0.5;  
reluc = 0;
a = 1;

%%%% weight initialization %%%%
W1 = rand(J,I) - 0.5;
W2 = rand(nos,J);
Wf = rand(nos,1).*exp(1i*2*pi*rand(nos,1));
Wfr = real(Wf);
Wfi = imag(Wf);
Wfm = abs(Wf);
Wfa = angle(Wf);
Wfmarr = zeros(nos,nepoch);
Wfaarr = zeros(nos,nepoch);

%%%% learning parameter %%%%
epsW1 = 0.0001;
epsW2 = 0.0001;
epsWfr = 0.000001;
epsWfi = 0.000001;


for ne = 1:nepoch
    ne 
    %%% forward propagation %%%
    Xh = sigmf(W1*X, [a 0]);
    omega = relu(W2*Xh,relum,reluc);
    
    
    r(:,1) = r0;
    phi(:,1) = phi0;
    for i = 1:N-1
        r(:,i+1) = r(:,i) + (r(:,i) - r(:,i).^3)*dt;
        phi(:,i+1) = phi(:,i) + omega*dt;    
    end
    Z = r.*exp(1i*phi);
    Y = real(sum(repmat(Wf,[1,N]).*Z));
    
    %%% backpropagation %%%
    delWfr = sum(repmat(O - Y,[nos,1]).*real(Z)*(-1),2);
    delWfi = sum(repmat(O - Y,[nos,1]).*imag(Z),2);
    delomgr = sum(repmat(Wfr,[1,N]).*sin(phi).*repmat((-1)*(O - Y),[nos,1]),2);
    delomgi = sum(repmat(Wfi,[1,N]).*cos(phi).*repmat((O - Y),[nos,1]),2);
    delomg = delomgr + delomgi;
    %delomg = abs(delomgr + 1i*delomgi);
    delW2 = delomg.*relugrad(omega,relum,reluc)*Xh';
    delW1 = (W2'*(delomg.*relugrad(omega,relum,reluc))).*(a*Xh.*(1-Xh))*X';
    W1 = W1 + epsW1*delW1;
    W2 = W2 + epsW2*delW2;
    Wfr = Wfr - epsWfr*delWfr;
    Wfi = Wfi - epsWfi*delWfi;
    Wf = Wfr + 1i*Wfi;
    
    Wfmarr(:,ne) = abs(Wf);
    Wfaarr(:,ne) = angle(Wf);
    omegaarr(:,ne) = omega;
    err(ne) = norm(O-Y);
    
end

figure(1)
plot(t,Y,'linewidth',1.4)
hold on
plot(t,O,'linewidth',1.4)
legend('Y_p','Y_d')
xlim([0 t(N)])
xlabel('time','Fontsize', 14, 'FontWeight','bold')
% 
figure(2)
subplot(3,1,1)
for i = 1:nos
    plot(eparr,omegaarr(i,:),'linewidth',1.4)
    hold on
    plot(eparr,ones(1,length(eparr))*omegaip(i),'--k')
end
ylabel('\omega','Fontsize', 14, 'FontWeight','bold')
subplot(3,1,2)
for i = 1:nos
    plot(eparr,Wfmarr(i,:),'linewidth',1.4)
    hold on
    plot(eparr,ones(1,length(eparr))*A(i),'--k')
end
ylabel('mod(Wf)','Fontsize', 14, 'FontWeight','bold')
subplot(3,1,3)
for i = 1:nos
    plot(eparr,Wfaarr(i,:),'linewidth',1.4)
    hold on
    plot(eparr,ones(1,length(eparr))*phiip(i),'--k')
end
ylabel('angle(Wf)','Fontsize', 14, 'FontWeight','bold')

figure(3)
plot(eparr,err,'linewidth',1.4)
ylabel('error','Fontsize', 14, 'FontWeight','bold')
xlabel('epoch no','Fontsize', 14, 'FontWeight','bold')