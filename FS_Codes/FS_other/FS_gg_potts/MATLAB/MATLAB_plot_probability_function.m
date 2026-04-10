clear;
close all;
clc;

T = -10+273.15;
k = 1.3806488e-23;

P0 = 0.5;
c = 5dege-20;

energy = -5:.01:5;
Pflip = zeros(size(energy));
Pflip(energy<=0) = P0;
Pflip(energy>0) = P0.*exp( (-energy(energy>0)*c)./(k*T) );

hold on;
plot(energy,Pflip,'k-');
box on
title(['c/(k*T) = ' num2str(c/(k*T))]);
plot([0,0],[0,P0],'--');
set(gca,'XTick',min(energy):1:max(energy));
hold off;

xlabel('\DeltaE');
ylabel('Pflip');