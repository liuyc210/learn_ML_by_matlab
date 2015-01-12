clear all;
n = 1e2;

% generate (x,y)
offset = 4.5;
RawData = (unidrnd(2,1,n)-1); %between 0 and 1
 RawData = [ones(1,n/2), zeros(1,n/2)];
 RawData = repmat([0,1], 1, n/2);
x = 0.5*(RawData-0.5) + normrnd(0,.0, 1, n) + offset;

% figure;plot(x,'.');

% training perceptor
NumIter = 1000;

theta0 = zeros(1,NumIter); 
theta1 = ones(1,NumIter)*1;
mu0 = 0.05;
mu1 = 0.05;
gradient0 = zeros(1,NumIter);
gradient1 = zeros(1,NumIter);
h = zeros(1,n);
for nIter = 2:NumIter
    for i=2:n
        h(i) = sigmoid(theta0(nIter) + theta1(nIter)*x(i));
        %     gradient(i) = RawData(i)*(1-h(i)) - (1-RawData(i))*h(i);
        gradient0(nIter) = gradient0(nIter) + (RawData(i) - h(i))*1;  %be care of the gradient
        gradient1(nIter) = gradient1(nIter) + (RawData(i) - h(i))*x(i);  %be care of the gradient
%         gradient0(nIter) = gradient0(nIter) + (RawData(i)-h(i))*h(i)*(1-h(i))*1;
%         gradient1(nIter) = gradient1(nIter) + (RawData(i)-h(i))*h(i)*(1-h(i))*x(i);
    end
    theta0(nIter+1) = theta0(nIter) + mu0*gradient0(nIter);
    theta1(nIter+1) = theta1(nIter) + mu1*gradient1(nIter);
end

[theta0(NumIter), theta1(NumIter)]
figure;
subplot(2,2,1); plot(theta0, '.'); hold on; plot(theta1,'r.'); grid on; title('theta');
subplot(2,2,2); plot(gradient0,'.'); hold on; plot(gradient1,'r.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');

x_axis = -10:0.1:10;
H = zeros(1,length(x_axis));
for i=1:length(x_axis) H(i) = sigmoid(theta0(NumIter)+theta1(NumIter)*x_axis(i)); end;
figure; plot(x_axis, H); grid on; 


