function [Positions] = Positions(Start,End, N)
Positions = [linspace(Start(1), End(1), N) ; linspace(Start(2), End(2), N) ; linspace(Start(3), End(3), N)];

end

