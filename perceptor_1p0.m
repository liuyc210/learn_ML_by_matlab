n = 1e2;

% generate (x,y)
RawData = 1*(unidrnd(2,1,n)-1); %between 0 and 1
x = 2*(RawData -0.5) + normrnd(0,0.8, 1, n);
RawData(1:10)  
% figure;plot(x,'.');  

% training perceptor

NumIter = 100;
theta = ones(1,NumIter); theta(1) = 11;
mu = 0.05;
gradient = zeros(1,NumIter);
h = zeros(1,NumIter);
for nIter = 2:NumIter
    for i=2:n
        h(i) = sigmoid(theta(nIter)*x(i));
%         gradient(nIter) = gradient(nIter) + (RawData(i) - h(i))*x(i);  %be care of the gradient
        gradient(nIter) = gradient(nIter) + (RawData(i) - h(i))*x(i)*h(i)*(1-h(i));  %be care of the gradient
    end
    theta(nIter+1) = theta(nIter) + mu*gradient(nIter);
end

figure;
subplot(2,2,1); plot(theta,'.'); grid on; title('theta');
subplot(2,2,2); plot(gradient,'.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');

nJ = 300;
J_theta = zeros(1,nJ);
for nTheta = 1:nJ
    for i=1:n
        J_theta(nTheta) = J_theta(nTheta) + (sigmoid((nTheta-nJ/2)/5*x(i)) - RawData(i))^2;
    end;
end
figure; plot(((1:nJ)-nJ/2)/5, J_theta); grid on; xlabel('theta'); ylabel('cost function');






