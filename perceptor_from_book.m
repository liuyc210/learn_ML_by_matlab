 P = [-.5 -.5 .3 -.1 -40; -.5 -5 -.5 1 50];
 T = [1 1 0 0 1];
 plotpv(P,T);
%  net = newp([-40 1; -1 50], 1);
net = newp([-40 1; -1 50], 1, 'hardlim', 'learnpn');
 hold on;
 linehandle = plotpc(net.IW{1}, net.b{1});
 E = 1;
 net.adaptParam.passes = 30;
 while (sse(E))
     [net, Y,E] = adapt(net, P, T);
     linehandle = plotpc(net.IW{1}, net.b{1}, linehandle);
     drawnow;
 end