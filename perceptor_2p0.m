clear all;
n = 1e4;

% generate (x,y)
offset = 1.5;
RawData = (unidrnd(2,1,n)-1); %between 0 and 1
% RawData = [ones(1,n/2), zeros(1,n/2)];
x = 2.5*(RawData-0.5) + normrnd(0,.0, 1, n) + offset;

% figure;plot(x,'.');

% training perceptor
theta0 = zeros(1,n); 
theta1 = ones(1,n)*1;
mu0 = 0.01;
mu1 = 0.03;
gradient0 = zeros(1,n);
gradient0 = zeros(1,n);
h = zeros(1,n);
for i=2:n
    h(i) = sigmoid(theta0(i) + theta1(i)*x(i));
%     gradient(i) = RawData(i)*(1-h(i)) - (1-RawData(i))*h(i);
    gradient0(i) = (RawData(i) - h(i))*1;  %be care of the gradient
    gradient1(i) = (RawData(i) - h(i))*x(i);  %be care of the gradient
    gradient0(i) = (RawData(i)-h(i))*h(i)*(1-h(i))*1;
    gradient1(i) = (RawData(i)-h(i))*h(i)*(1-h(i))*x(i);

        theta0(i+1) = theta0(i) + mu0*gradient0(i);
        theta1(i+1) = theta1(i) + mu1*gradient1(i);
%     end
end

[theta0(n), theta1(n)]
figure;
subplot(2,2,1); plot(theta0, '.'); hold on; plot(theta1,'r.'); grid on; title('theta');
subplot(2,2,2); plot(gradient0,'.'); hold on; plot(gradient1,'r.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');

x_axis = -10:0.1:10;
H = zeros(1,length(x_axis));
for i=1:length(x_axis) H(i) = sigmoid(theta0(n)+theta1(n)*x_axis(i)); end;
figure; plot(x_axis, H); grid on; 


