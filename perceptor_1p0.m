n = 1e3;

% generate (x,y)
RawData = 1*(unidrnd(2,1,n)-1); %between 0 and 1
x = 2*(RawData -0.5) + normrnd(0,0.2, 1, n);
RawData(1:10)
% figure;plot(x,'.');

% training perceptor
theta = ones(1,n); theta(1) = 1;
mu = 0.05;
gradient = zeros(1,n);
h = zeros(1,n);
for i=2:n
    h(i) = sigmoid(theta(i)*x(i));
%     gradient(i) = RawData(i)*(1-h(i)) - (1-RawData(i))*h(i);
    gradient(i) = (RawData(i) - h(i))*x(i);  %be care of the gradient
    theta(i+1) = theta(i) + mu*gradient(i);
end

figure;
subplot(2,2,1); plot(theta,'.'); grid on; title('theta');
subplot(2,2,2); plot(gradient,'.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');




