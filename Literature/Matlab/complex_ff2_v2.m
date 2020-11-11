clear all;
close all;

%%% desired O/P %%%
load('gait_patterns.mat')
gaitp1 = gaitp(:,:,1:1000);
t = t(1:1000);
dt = 0.001;

dimns = size(gaitp1);
O = zeros(dimns);
Yp = zeros(dimns);
N = dimns(3);
for i = 1:dimns(1)
    for j = 1:dimns(2)
        O(i,j,:) = gaitp1(i,j,:) - mean(gaitp1(i,j,:));
        O(i,j,:) = O(i,j,:)/(1.2*max(abs(O(i,j,:))));
    end
end
T = N*dt;

%%% desired input %%%
Kbs = 100;    %input dimension
dip = zeros(dimns(1),Kbs);
sigma = 5;
for i = 1:dimns(1)
    dip(i,:) = exp(-((linspace(1,Kbs,Kbs) - (i-0.92)*floor(Kbs/(dimns(1)-1)))/sigma).^2); %- 1i*exp(-((linspace(1,Kbs,Kbs) - i*floor(Kbs/dimns(1)))/sigma).^2);
end

%%Frequency specifications:
fs = 1/dt;
Fs = fs;
dF = Fs/N;                  % hertz
f = -Fs/2:dF:Fs/2-dF;           % hertz
i = 4;
O1_f = fftshift(fft(squeeze(O(i,1,:))));
O2_f = fftshift(fft(squeeze(O(i,2,:))));
figure(1)
subplot(2,1,1)
plot(t,squeeze(O(i,1,:)),'linewidth',1.5)
hold on
plot(t,squeeze(O(i,2,:)),'linewidth',1.5)
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
funfreq = zeros(1,dimns(1));
for i = 1:dimns(1)
    [peaks, locations] = findpeaks(squeeze(O(i,1,:)));
    funfreq(i) = 1/((locations(2) - locations(1))*dt);
end

%%% second phase of training %%%

%%% oscillator veriables %%%
Kcpg = 10;
omega = zeros(dimns(1),Kcpg);
for i = 1:dimns(1)
    omega(i,:) = linspace(1,Kcpg,Kcpg)'*(funfreq(i)*2*pi);
end
r = zeros(Kcpg,N);
phi = zeros(Kcpg,N);
Zr = zeros(Kcpg,N);
Zi = zeros(Kcpg,N);

%%% CPG complex MLP network architecture %%%
Lcpg = 20;  % hiddenlayer dimension
Mcpg = dimns(2);  % output dimension
ahcpg = 0.5;
aocpg = 0.5;



%%% learning parameter %%%
etaW1cpg = 0.001;
etaW2cpg = 0.001;
etaW1bs = 0.001;
etaW2bs = 0.001;


nepochscpg = 10000; %no of epochs
error = zeros(1,nepochscpg);


for ip = 1:dimns(1)
    
    W1cpgr = rand(Lcpg,Kcpg);
    W1cpgi = rand(Lcpg,Kcpg);
    W1cpg = W1cpgr + 1i*W1cpgi;
    W2cpgr = rand(Mcpg,Lcpg);
    W2cpgi = rand(Mcpg,Lcpg);
    W2cpg = W2cpgr + 1i*W2cpgi;
    
    r(:,1) = ones(Kcpg,1);
    phi(:,1) = zeros(Kcpg,1);
    Zr(:,1) = real(r(:,1).*exp(1i*phi(:,1)));
    Zi(:,1) = imag(r(:,1).*exp(1i*phi(:,1)));
    for it = 1:N-1
        r(:,it+1) = r(:,it) + ((1 - r(:,it).^2).*r(:,it))*dt;
        phi(:,it+1) = phi(:,it) + omega(ip,:)'*dt;
        Zr(:,it+1) = real(r(:,it+1).*exp(1i*phi(:,it+1)));
        Zi(:,it+1) = imag(r(:,it+1).*exp(1i*phi(:,it+1)));
    end
    Z = Zr + 1i*Zi;
    
    for nep = 1:nepochscpg
        nep
        
        %%% forward propagation through complex MLP %%%
        %     X = X(:,ip);
        %     O = O(:,ip);
        nh = W1cpg*Z;
        nhr = real(nh); nhi = imag(nh);
        xhr = 2*sigmf(nhr,[ahcpg,0]) - 1; xhi = 2*sigmf(nhi,[ahcpg,0]) - 1;
        xh = xhr + 1i*xhi;
        no = W2cpg*xh;
        nor = real(no); noi = imag(no);
        yr = 2*sigmf(nor,[aocpg,0]) - 1; yi = 2*sigmf(noi,[aocpg,0]) - 1;
        y = yr + 1i*yi;
        Y = yr;
        
        
        %%% error computation in complex MLP %%%
        error(nep) = norm(squeeze(O(ip,:,:)) - Y);
        
        %%% backpropagation through complex MLP %%%
        dW2cpgr = (-1)*((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))*xhr';
        dW2cpgi = ((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))*xhi';
        dW1cpgr = (-1)*((W2cpgr'*((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))).*((ahcpg/2)*(1+xhr).*(1-xhr)))*real(Z)' ...
            + ((W2cpgi'*((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))).*((ahcpg/2)*(1+xhi).*(1-xhi)))*imag(Z)';
        dW1cpgi = ((W2cpgr'*((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))).*((ahcpg/2)*(1+xhr).*(1-xhr)))*imag(Z)' ...
            + ((W2cpgi'*((squeeze(O(ip,:,:)) - Y).*((aocpg/2)*(1+yr).*(1-yr)))).*((ahcpg/2)*(1+xhi).*(1-xhi)))*real(Z)';
        
        %%% updating weights of complex MLP %%%
        W2cpgr = W2cpgr - etaW2cpg*dW2cpgr;
        W2cpgi = W2cpgi - etaW2cpg*dW2cpgi;
        W2cpg = W2cpgr + 1i*W2cpgi;
        W1cpgr = W1cpgr - etaW1cpg*dW1cpgr;
        W1cpgi = W1cpgi - etaW1cpg*dW1cpgi;
        W1cpg = W1cpgr + 1i*W1cpgi;
        
    end
    Yp(ip,:,:) = Y;
    Wcpg(ip,:) = cat(1,W1cpg(:),W2cpg(:));
end

k = 1;
figure(1)
for i = 1:dimns(1)
    for j = 1:dimns(2)
        subplot(dimns(1),dimns(2),k)
        plot(t,squeeze(O(i,j,:)))
        hold on
        plot(t,squeeze(Yp(i,j,:)))
        k = k+1;
    end 
end

Wcpgrmax = max(max(abs(real(Wcpg))));
Wcpgimax = max(max(abs(imag(Wcpg))));
Wcpgnorm = real(Wcpg)/Wcpgrmax + 1i*imag(Wcpg)/Wcpgimax;


%%% population vector to CPG parameter complex MLP or brainstem complex MLP architecture %%%
Lbs = 2000;%Kcpg*Lcpg+Lcpg*Mcpg+20;     %hidden layer dimension 
Mbs = Kcpg*Lcpg+Lcpg*Mcpg;      % output dimension
ahbs = 0.5;
aobs = 0.5;
Wcpg = zeros(dimns(1),Mbs);

W1bsr = rand(Lbs,Kbs);
W1bsi = rand(Lbs,Kbs);
W1bs = W1bsr + 1i*W1bsi;
W2bsr = rand(Mbs,Lbs);
W2bsi = rand(Mbs,Lbs);
W2bs = W2bsr + 1i*W2bsi;

nepochsbs = 20000; %no of epochs
error1 = zeros(1,nepochsbs);

for nep1 = 1:nepochsbs
    nep1
    for ip = 1:dimns(1)
        %%% Forward propagation %%%
        nhbs = W1bs*dip(ip,:)';
        nhbsr = real(nhbs); nhbsi = imag(nhbs);
        xhbsr = 2*sigmf(nhbsr,[ahbs,0]) - 1; xhbsi = 2*sigmf(nhbsi,[ahbs,0]) - 1;
        xhbs = xhbsr + 1i*xhbsi;
        nobs = W2bs*xhbs;
        nobsr = real(nobs); nobsi = imag(nobs);
        ybsr = 2*sigmf(nobsr,[aobs,0]) - 1; ybsi = 2*sigmf(nobsi,[aobs,0]) - 1;
        ybs = ybsr + 1i*ybsi;
        
        error1(nep1) = error1(nep1) + norm(Wcpgnorm(ip,:) - ybs');
        
        dW2bsr = (-1)*((real(Wcpgnorm(ip,:)') - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsr' ...
            + (-1)*((imag(Wcpgnorm(ip,:)') - ybsi).*((aobs/2)*(1+ybsi).*(1-ybsi)))*xhbsi';
        dW2bsi = ((real(Wcpgnorm(ip,:)') - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsi' ...
            + (-1)*((imag(Wcpgnorm(ip,:)') - ybsi).*((aobs/2)*(1+ybsi).*(1-ybsi)))*xhbsr';
        dW1bsr = (-1)*((W2bsr'*((real(Wcpgnorm(ip,:)') - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*dip(ip,:) ...
            + (-1)*((W2bsi'*((imag(Wcpgnorm(ip,:)') - ybsi).*((aobs/2)*(1+ybsi).*(1-ybsi)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*dip(ip,:);
        dW1bsi = ((W2bsi'*((real(Wcpgnorm(ip,:)') - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*dip(ip,:) ...
            + (-1)*((W2bsr'*((imag(Wcpgnorm(ip,:)') - ybsi).*((aobs/2)*(1+ybsi).*(1-ybsi)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*dip(ip,:);
            
        %%% updating weights of complex MLP %%%
        W2bsr = W2bsr - etaW2bs*dW2bsr;
        W2bsi = W2bsi - etaW2bs*dW2bsi;
        W2bs = W2bsr + 1i*W2bsi;
        W1bsr = W1bsr - etaW1bs*dW1bsr;
        W1bsi = W1bsi - etaW1bs*dW1bsi;
        W1bs = W1bsr + 1i*W1bsi;
    end
end
figure(2)
subplot(3,1,1)
plot(t,squeeze(O(1,1,:)),'linewidth',1.4)
hold on 
plot(t,Y(1,:),'linewidth',1.4)
legend('O_1(t)','Y_1(t)')
xlim([0 t(N)])
subplot(3,1,2)
plot(t,squeeze(O(1,2,:)),'linewidth',1.4)
hold on 
plot(t,Y(2,:),'linewidth',1.4)
legend('O_2(t)','Y_2(t)')
xlabel('Time (sec)', 'Fontsize', 12,'FontWeight','bold');
xlim([0 t(N)])
subplot(3,1,3)
plot(error,'linewidth',1.4)
xlabel('No of epochs', 'Fontsize', 12,'FontWeight','bold');
ylabel('error', 'Fontsize', 12,'FontWeight','bold');
