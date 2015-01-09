clear all;
n = 1e5;

% generate (x,y)
offset = 1.5;
x = normrnd(0,1,1,n);
for i=1:length(x) 
    if (abs(x(i))<1) 
        x(i) = x(i)*0.3; 
    else
        x(i) = x(i)*3; 
    end; 
end;
x = 3*ones(1,n);
RawData = (abs(x)>1)*1; %between 0 and 1
% figure; plot(x,'.'); hold on; plot(RawData,'r.'); grid on;

% figure;plot(x,'.');

% training perceptor
t1_11 = ones(1,n)*(3);  %theta
t1_12 = ones(1,n)*3; 
t1_21 = ones(1,n)*3; 
t1_22 = ones(1,n)*-3; 
t2_01 = ones(1,n)*-2; 
t2_11 = ones(1,n)*2; 
t2_21 = ones(1,n)*2; 
mu = 0.03;
g1_11 = zeros(1,n); %gradient
g1_12 = zeros(1,n); 
g1_21 = zeros(1,n); 
g1_12 = zeros(1,n); 
g2_01 = zeros(1,n); 
g2_11 = zeros(1,n); 
g2_21 = zeros(1,n); 

e2 = zeros(1,n); %error
e1_1 = zeros(1,n);
e1_2 = zeros(1,n);

o1 = zeros(1,n); %output of hidden
o2 = zeros(1,n);

h = zeros(1,n);    %result of forward propagation
for i=2:n
    %get forward propagation
    o1(i) = sigmoid(1*t1_11(i) + x(i)*t1_21(i));
    o2(i) = sigmoid(1*t1_12(i) + x(i)*t1_22(i));
    h(i) =  sigmoid(...
            t2_01(i)*1 + ...
            t2_11(i)*o1(i) + ...
            t2_21(i)*o2(i) ); 
                                        %     sigmoid(theta0(i) + theta1(i)*x(i));
    e2(i) = (RawData(i)-h(i))*h(i)*(1-h(i));

%     e1_2(i) = e2(i)*t2_21(i)*h(i)*(1-h(i));
%      e1_1(i) = e2(i)*t2_11(i)*h(i)*(1-h(i));
     e1_2(i) = e2(i)*t2_21(i)*o1(i)*(1-o1(i));
     e1_1(i) = e2(i)*t2_11(i)*o2(i)*(1-o2(i));
    
    t2_01(i+1) = t2_01(i) + mu*e2(i)*1;
    t2_11(i+1) = t2_11(i) + mu*e2(i)*o1(i);
    t2_21(i+1) = t2_21(i) + mu*e2(i)*o2(i);
    
    t1_11(i+1) = t1_11(i) + mu*e1_1(i)*1;
    t1_21(i+1) = t1_21(i) + mu*e1_1(i)*x(i);
    t1_12(i+1) = t1_12(i) + mu*e1_2(i)*1;
    t1_22(i+1) = t1_22(i) + mu*e1_2(i)*x(i);

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


x_axis = -10:0.1:10;
H = zeros(1,length(x_axis));
for i=1:length(x_axis)
    o1(i) = sigmoid(1*t1_11(n) + x_axis(i)*t1_21(n));
    o2(i) = sigmoid(1*t1_12(n) + x_axis(i)*t1_22(n));
%     H(i) =  sigmoid(   t2_01(n)*1 +   t2_11(n)*o1(i) +  t2_21(n)*o2(i) ); 
    H(i) =  sigmoid(   t2_01(n)*1 +   t2_11(n)*o1(i) +  t2_21(n)*o2(i) ); 
end;
figure; title('final H');
subplot(3,1,1); plot(x_axis, H); grid on; 
subplot(3,1,2); plot(x_axis, o1(1:length(H))); grid on; 
subplot(3,1,3); plot(x_axis, o2(1:length(H))); grid on; 

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

