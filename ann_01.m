clear all;
n = 2e2;
NumIter = 9e2;

% generate (x,y)
offset = 1.5;
x = (unidrnd(20,1,n)-10);
RawData = (abs(x-1)<4)*1; %between 0 and 1
% figure; plot(x,'.'); hold on; plot(RawData,'r.'); grid on;



% training perceptor
t1_11 = ones(1,NumIter)*(1);  %theta
t1_12 = ones(1,NumIter)*1; 
t1_21 = ones(1,NumIter)*5; 
t1_22 = ones(1,NumIter)*-5; 
t2_01 = ones(1,NumIter)*-2; 
t2_11 = ones(1,NumIter)*2; 
t2_21 = ones(1,NumIter)*2; 
mu = 0.07;
g1_11 = zeros(1,NumIter); %gradient
g1_12 = zeros(1,NumIter); 
g1_21 = zeros(1,NumIter); 
g1_12 = zeros(1,NumIter);
g2_01 = zeros(1,NumIter); close all;
g2_11 = zeros(1,NumIter); 
g2_21 = zeros(1,NumIter); 

e2 = zeros(1,n); %error
e1_1 = zeros(1,n);
e1_2 = zeros(1,n);

o1 = zeros(1,n); %output of hidden
o2 = zeros(1,n);

h = zeros(1,n);    %result of forward propagation
for nIter = 2:NumIter-1
    for i=1:n
        %get forward propagation
        o1(i) = sigmoid(1*t1_11(nIter) + x(i)*t1_21(nIter));
        o2(i) = sigmoid(1*t1_12(nIter) + x(i)*t1_22(nIter));
        h(i) =  sigmoid(...
            t2_01(nIter)*1 + ...
            t2_11(nIter)*o1(i) + ...
            t2_21(nIter)*o2(i) );
        %     sigmoid(theta0(i) + theta1(i)*x(i));
        e2(i) = (RawData(i)-h(i))*h(i)*(1-h(i));
        
        %     e1_2(i) = e2(i)*t2_21(i)*h(i)*(1-h(i));
        %      e1_1(i) = e2(i)*t2_11(i)*h(i)*(1-h(i));
        e1_2(i) = e2(i)*t2_21(nIter)*o2(i)*(1-o2(i));  %there was a error here
        e1_1(i) = e2(i)*t2_11(nIter)*o1(i)*(1-o1(i));  %there was a error here
    end
    
        t2_01(nIter+1) = t2_01(nIter) + mu*sum(e2*1);
        t2_11(nIter+1) = t2_11(nIter) + mu*dot(e2, o1);
        t2_21(nIter+1) = t2_21(nIter) + mu*dot(e2, o2);
        
        t1_11(nIter+1) = t1_11(nIter) + mu*sum(e1_1*1);
        t1_21(nIter+1) = t1_21(nIter) + mu*dot(e1_1,x);
        t1_12(nIter+1) = t1_12(nIter) + mu*sum(e1_2*1);
        t1_22(nIter+1) = t1_22(nIter) + mu*dot(e1_2, x);
%         
end

figure;
subplot(2,2,1); plot(o1, '.'); %hold on; plot(theta1,'r.'); grid on; title('theta');
subplot(2,2,2); plot(o2,'.'); %hold on; plot(gradient1,'r.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');

figure;
subplot(2,2,1); plot(t1_11, '.'); grid on; title('t1\_11');
subplot(2,2,2); plot(t1_12,'.'); grid on; title('t1\_12');
subplot(2,2,3); plot(t1_21,'.'); grid on; title('t1\_21');
subplot(2,2,4); plot(t1_22,'.'); grid on; title('t1\_22');

figure;
subplot(3,1,1); plot(t2_01, '.'); grid on; title('t2\_01');
subplot(3,1,2); plot(t2_11,'.'); grid on; title('t2\_11');
subplot(3,1,3); plot(t2_21,'.'); grid on; title('t2\_21');



x_axis = linspace(-10, 10, 100);
H = zeros(1,100);
for i=1:length(x_axis)
    ko1(i) = sigmoid(1*t1_11(nIter) + x_axis(i)*t1_21(nIter));
    ko2(i) = sigmoid(1*t1_12(nIter) + x_axis(i)*t1_22(nIter));
%     H(i) =  sigmoid(   t2_01(n)*1 +   t2_11(n)*o1(i) +  t2_21(n)*o2(i) ); 
    H(i) =  sigmoid(   t2_01(nIter)*1 +   t2_11(nIter)*ko1(i) +  t2_21(nIter)*ko2(i) ); 
end;
figure; title('final H');
subplot(3,1,1); plot(x_axis, H); grid on; 
subplot(3,1,2); plot(x_axis, ko1(1:length(H))); grid on; 
subplot(3,1,3); plot(x_axis, ko2(1:length(H))); grid on; 

break;
x_axis = -10:0.1:10;
H = zeros(1,length(x_axis));
for i=1:length(x_axis)
    o1(i) = sigmoid(1*t1_11(3) + x_axis(i)*t1_21(3));
    o2(i) = sigmoid(1*t1_12(3) + x_axis(i)*t1_22(3));
%     H(i) =  sigmoid(   t2_01(n)*1 +   t2_11(n)*o1(i) +  t2_21(n)*o2(i) ); 
    H(i) =  sigmoid(   t2_01(3)*1 +   t2_11(3)*o1(i) +  t2_21(3)*o2(i) ); 
end;
figure; title('init H');
subplot(3,1,1); plot(x_axis, H); grid on; 
subplot(3,1,2); plot(x_axis, o1(1:length(H))); grid on; 
subplot(3,1,3); plot(x_axis, o2(1:length(H))); grid on; 

