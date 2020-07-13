clear all
close all

T = 2*pi/5;
dt = 0.001;
N = floor(T/dt);
t = (0:N-1)*dt;
nepoch = 3000;
eparr = 1:nepoch;

%%%% Network architecture %%%%
nos = 3; % no of oscillators
J = 10;   % no of hidden layer neurons in the I/P MLP
I = 3;   % no of I/P nodes

%%%% desired I/P-O/P %%%%
P = 3; % no of I/P-O/P patterns
X = rand(I,P);
O = zeros(P,N);
omegaip = 2*pi*(2*rand(nos,P));
for p = 1:P
    for i =1:nos
        O(p,:) = O(p,:) + cos(omegaip(i,p)*t);
    end
end
Y = zeros(P,N);
r = zeros(nos,N);
phi = zeros(nos,N);
r0 = ones(nos,1);
phi0 = zeros(nos,1);
omegaarr = zeros(nos,P,nepoch);
err = zeros(1,nepoch);

%%%% activation function parameter %%%%
% parameter of I/P MLP activation function
relum = 0.5;  
reluc = 0;
a = 1;

%%%% weight initialization %%%%
W1 = rand(J,I) - 0.5;
W2 = rand(nos,J);

%%%% learning parameter %%%%
epsW1 = 0.0005;
epsW2 = 0.0005;


for ne = 1:nepoch
    ne
    for p = 1:P
        %%% forward propagation %%%
        H = W1*X(:,p);
        Xh = sigmf(H, [a 0]);
        Homg = W2*Xh;
        omega = relu(Homg,relum,reluc);
        omegaarr(:,p,ne) = omega;
        
        r(:,1) = r0;
        phi(:,1) = phi0;
        for i = 1:N-1
            r(:,i+1) = r(:,i) + (r(:,i) - r(:,i).^3)*dt;
            phi(:,i+1) = phi(:,i) + omega*dt;
        end
        Y(p,:) = sum(r.*cos(phi));
        
        %%% backpropagation %%%
        delomg = sum(sin(phi).*repmat(O(p,:) - Y(p,:),[nos,1]),2);
        delW2 = delomg.*relugrad(omega,relum,reluc)*Xh';
        delW1 = (W2'*(delomg.*relugrad(omega,relum,reluc))).*(a*Xh.*(1-Xh))*X(:,p)';
        W1 = W1 - epsW1*delW1;
        W2 = W2 - epsW2*delW2;
        
    end
    err(ne) = norm(O-Y);
end

figure(1)
for p =1:P
    subplot(P,1,p)
    plot(t,Y(p,:),'linewidth',1.4)
    hold on
    plot(t,O(p,:),'linewidth',1.4)
    legend('Y_d','Y_p')
    xlim([0 t(N)])
    ylabel(strcat('Y_',int2str(p)),'Fontsize', 14, 'FontWeight','bold')
end
xlabel('time (sec)','Fontsize', 14, 'FontWeight','bold')
% 
figure(2)
for p = 1:P
    subplot(P,1,p)
    for i = 1:nos
        plot(eparr,squeeze(omegaarr(i,p,:)),'linewidth',1.4)
        hold on
    end
    ylabel('\omega','Fontsize', 14, 'FontWeight','bold')
end
xlabel('epoch no','Fontsize', 14, 'FontWeight','bold')

figure(3)
plot(eparr,err,'linewidth',1.4)
ylabel('error','Fontsize', 14, 'FontWeight','bold')
xlabel('epoch no','Fontsize', 14, 'FontWeight','bold')