function s = steervector(Geometry, Pars, doas)

Beta = 2*pi/Pars.lambda;

ar = [sin(doas(2))*cos(doas(1)) sin(doas(2))*sin(doas(1)) cos(doas(2))];
ar = normalize(ar);

r = zeros(3,Geometry.Nant^2);

for ant = 1:Geometry.Nant^2
    
    r(:,ant) = [Geometry.BSPos(1)+Geometry.BSAntennaPos(1,ant) Geometry.BSPos(2)+Geometry.BSAntennaPos(2,ant) Geometry.BSPos(3)+Geometry.BSAntennaPos(3,ant)];
    r(:,ant) = normalize(r(:,ant));
    s(ant) = exp(1i*ar*r(:,ant));
end

s = s.';


end
