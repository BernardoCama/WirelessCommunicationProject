function [distance] = dist3D(s,d)
distance = sqrt(sum(s(1,1:3)-d(1,1:3)).^2);
end

