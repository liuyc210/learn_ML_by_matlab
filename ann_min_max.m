nSample = 100;

x = normrnd(0,10,nSample, 2);
for i=1:nSample y(i) = x(i,1)<x(i,2); end;

nIter = 1000;
w1 = ones(1, nIter);
w2 = -w1;
out = zeros(1,nSample);
mu1 = 0.01;
mu2 = 0.01;

figure; 

for i=2:nIter
    g1 = 0;
    g2 = g1;
    for k=1:nSample
        out(k) = sigmoid(w1(i)*x(k,1) + w2(i)*x(k,2));
        g1 = g1 + (y(k) - out(k))*x(k,1)*out(k)*(1-out(k)); 
        g2 = g2 + (y(k) - out(k))*x(k,2)*out(k)*(1-out(k)); 
    end
    w1(i+1) = w1(i) + mu1*g1;
    w2(i+1) = w2(i) + mu2*g2;
    if i==2 subplot(2,2,2); plot(out,'.'); grid on; end
end

subplot(2,2,1); plot(w1); hold on; plot(w2,'r'); grid on;
subplot(2,2,3); plot(out,'.'); grid on;