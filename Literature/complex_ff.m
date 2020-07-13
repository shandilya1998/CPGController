clear all;
close all;

T = 6*pi/2;
dt = 0.001;
N = floor(T/dt);
t = (0:N-1)*dt;

%%% desired O/P %%%
P = 1;
mags = 1.5*rand(3,2);
phoff = 2*pi*rand(3,2);
O = zeros(2,N);
O(1,:) = mags(1,1)*cos(2*t + phoff(1,1)) + mags(2,1)*cos(4*t + phoff(2,1)) + mags(3,1)*cos(6*t + phoff(3,1));
O(2,:) = mags(1,2)*cos(2*t + phoff(1,2)) + mags(2,2)*cos(4*t + phoff(2,2)) + mags(3,2)*cos(6*t + phoff(3,2));
O(1,:) = O(1,:)/(1.2*max(O(1,:)));
O(2,:) = O(2,:)/(1.2*max(O(2,:)));

%%% oscillator veriables %%%
nos = 3;
omega = [2;4;6];
r = zeros(nos,N);
phi = zeros(nos,N);
Zr = zeros(nos,N);
Zi = zeros(nos,N);

%%% complex MLP parameters %%%
I = nos;  % input dimension
J = 20;  % hiddenlayer dimension
K = 2;  % output dimension
W1r = rand(J,I);
W1i = rand(J,I);
W1 = W1r + 1i*W1i;
W2r = rand(K,J);
W2i = rand(K,J);
W2 = W2r + 1i*W2i;
ah = 0.5;
ao = 0.5;

%%% learning parameter %%%
etaW1 = 0.001;
etaW2 = 0.001;
nepochs = 3000; %no of epochs
error = zeros(1,nepochs);

r(:,1) = ones(nos,1);
phi(:,1) = zeros(nos,1);
Zr(:,1) = real(r(:,1).*exp(1i*phi(:,1)));
Zi(:,1) = imag(r(:,1).*exp(1i*phi(:,1)));
for it = 1:N-1
    r(:,it+1) = r(:,it) + ((1 - r(:,it).^2).*r(:,it))*dt;
    phi(:,it+1) = phi(:,it) + omega*dt;
    Zr(:,it+1) = real(r(:,it+1).*exp(1i*phi(:,it+1)));
    Zi(:,it+1) = imag(r(:,it+1).*exp(1i*phi(:,it+1)));
end
X = Zr + 1i*Zi;

for nep = 1:nepochs
    nep
    
    %%% forward propagation through complex MLP %%%
%     X = X(:,ip);
%     O = O(:,ip);
    nh = W1*X;
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
    dW1r = (-1)*((W2r'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhr).*(1-xhr)))*real(X)' ...
              + ((W2i'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhi).*(1-xhi)))*imag(X)';
    dW1i = ((W2r'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhr).*(1-xhr)))*imag(X)' ...
         + ((W2i'*((O - Y).*((ao/2)*(1+yr).*(1-yr)))).*((ah/2)*(1+xhi).*(1-xhi)))*real(X)';
          
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
