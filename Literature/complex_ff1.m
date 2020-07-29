clear all;
close all;

%%% desired O/P %%%
M = 2;
a = csvread('gait_data.csv',1,0);
dt = 0.001;
asize = size(a);
nperiod = 5;    % no of periods of the desired O/P signal
pks = zeros(nperiod+1,8);
locs = zeros(nperiod+1,8);
lens = zeros(1,8);
for i = 3:asize(2)
    [pk, loc] = findpeaks(a(:,i));
    pks(:,i-2) = pk(1:nperiod+1);
    locs(:,i-2) = loc(1:nperiod+1);
    lens(i-2) = loc(nperiod+1) - loc(1);
end
N = lens(1);
O1 = zeros(8,N);
for i = 3:asize(2)
    a1 = a(locs(1,i-2)+1:locs(nperiod+1,i-2),i) - mean(a(locs(1,i-2)+1:locs(nperiod+1,i-2),i));
    a2 = a1/(1.2*max(abs(a1)));
    O1(i-2,:) = a2;
end
O = O1(1:M,:);
t = (0:N-1)*dt;
T = N*dt;

%%Frequency specifications:
fs = 1/dt;
Fs = fs;
dF = Fs/N;                  % hertz
f = -Fs/2:dF:Fs/2-dF;           % hertz
O1_f = fftshift(fft(O(1,:)));
O2_f = fftshift(fft(O(2,:)));
figure(1)
subplot(2,1,1)
plot(t,O(1,:),'linewidth',1.5)
hold on
plot(t,O(2,:),'linewidth',1.5)
xlabel('Time (Secs)', 'Fontsize', 15,'FontWeight','bold');
subplot(2,1,2)
plot(f,abs(O1_f)/N,'linewidth',1.5)
hold on
plot(f,abs(O2_f)/N,'linewidth',1.5)
xlim([-30 30])
legend('O1(t)','O2(t)')
xlabel('Frequency (in hertz)', 'Fontsize', 15,'FontWeight','bold');
title('Magnitude Response');

%%%% Finding fundamental frequency of O/P signals
O1_magspec = abs(O1_f)/N;
[peaks locations] = findpeaks(O1_magspec);
O1_fundf = f(locations((length(locations)+1)/2+1));

%%% second phase of training %%%

%%% oscillator veriables %%%
K = 20;
omega = linspace(1,K,K)'*(O1_fundf*2*pi);
r = zeros(K,N);
phi = zeros(K,N);
Zr = zeros(K,N);
Zi = zeros(K,N);

%%% complex MLP parameters %%%
L = 50;  % hiddenlayer dimension
M = 2;  % output dimension
W1r = rand(L,K);
W1i = rand(L,K);
W1 = W1r + 1i*W1i;
W2r = rand(M,L);
W2i = rand(M,L);
W2 = W2r + 1i*W2i;
ah = 0.5;
ao = 0.5;

%%% learning parameter %%%
etaW1 = 0.001;
etaW2 = 0.001;
nepochs = 3000; %no of epochs
error = zeros(1,nepochs);

r(:,1) = ones(K,1);
phi(:,1) = zeros(K,1);
Zr(:,1) = real(r(:,1).*exp(1i*phi(:,1)));
Zi(:,1) = imag(r(:,1).*exp(1i*phi(:,1)));
for it = 1:N-1
    r(:,it+1) = r(:,it) + ((1 - r(:,it).^2).*r(:,it))*dt;
    phi(:,it+1) = phi(:,it) + omega*dt;
    Zr(:,it+1) = real(r(:,it+1).*exp(1i*phi(:,it+1)));
    Zi(:,it+1) = imag(r(:,it+1).*exp(1i*phi(:,it+1)));
end
Z = Zr + 1i*Zi;

for nep = 1:nepochs
    nep
    
    %%% forward propagation through complex MLP %%%
%     X = X(:,ip);
%     O = O(:,ip);
    nh = W1*Z;
    nhr = real(nh); nhi = imag(nh);
    xhr = 2*sigmf(nhr,[ah,0]) - 1; xhi = 2*sigmf(nhi,[ah,0]) - 1;
    xh = xhr + 1i*xhi;
    no = W2*xh;
    nor = real(no); noi = imag(no);
    yr = 2*sigmf(nor,[ao,0]) - 1; yi = 2*sigmf(noi,[ao,0]) - 1;
    y = yr + 1i*yi;
    Y = yr;
    
    
    %%% error computation in complex MLP %%%
    error(nep) = norm(O - Y);
    
    %%% backpropagation through complex MLP %%%
    dW2r = (-1)*((O - Y).*((ao/2)*(1+yr).*(1-yr)))*xhr';
    dW2i = ((O - Y).*((ao/2)*(1+yr).*(1-yr)))*xhi';
    dW1r = (-1)*((W2r'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhr).*(1-xhr)))*real(Z)' ...
              + ((W2i'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhi).*(1-xhi)))*imag(Z)';
    dW1i = ((W2r'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhr).*(1-xhr)))*imag(Z)' ...
         + ((W2i'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhi).*(1-xhi)))*real(Z)';
          
    %%% updating weights of complex MLP %%%
    W2r = W2r - etaW2*dW2r;
    W2i = W2i - etaW2*dW2i;
    W2 = W2r + 1i*W2i;
    W1r = W1r - etaW1*dW1r;
    W1i = W1i - etaW1*dW1i;
    W1 = W1r + 1i*W1i;

end

figure(1)
subplot(3,1,1)
plot(t,O(1,:),'linewidth',1.4)
hold on 
plot(t,Y(1,:),'linewidth',1.4)
legend('O_1(t)','Y_1(t)')
xlim([0 t(N)])
subplot(3,1,2)
plot(t,O(2,:),'linewidth',1.4)
hold on 
plot(t,Y(2,:),'linewidth',1.4)
legend('O_2(t)','Y_2(t)')
xlabel('Time (sec)', 'Fontsize', 12,'FontWeight','bold');
xlim([0 t(N)])
subplot(3,1,3)
plot(error,'linewidth',1.4)
xlabel('No of epochs', 'Fontsize', 12,'FontWeight','bold');
ylabel('error', 'Fontsize', 12,'FontWeight','bold');
