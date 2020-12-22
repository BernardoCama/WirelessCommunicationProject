function fig = PlotScenario(Geometry)

fig = figure()
hold on

% Plot Antennas of Base Station
plt = zeros(1,Geometry.Nant^2);
for ant = 1:Geometry.Nant^2
    if ant == floor(Geometry.Nant^2/2)
        plt(ant) = plot3(Geometry.BSPos(1)+Geometry.BSAntennaPos(1,ant),Geometry.BSPos(2)+Geometry.BSAntennaPos(2,ant),Geometry.BSPos(3)+Geometry.BSAntennaPos(3,ant), 'Marker', 'v', 'Color', 'b', 'DisplayName', 'BS', 'LineStyle', 'none','MarkerSize', 15);
    else
        plt(ant) = plot3(Geometry.BSPos(1)+Geometry.BSAntennaPos(1,ant),Geometry.BSPos(2)+Geometry.BSAntennaPos(2,ant),Geometry.BSPos(3)+Geometry.BSAntennaPos(3,ant), 'Marker', 'v', 'Color', 'b', 'LineStyle', 'none', 'MarkerSize', 15);
    end

end

% Plot Pole of BS
plt0 = plot3([Geometry.BSPos(1) 0],[Geometry.BSPos(2) 0],[Geometry.BSPos(3) 0],'-', 'Color','b');


% Plot Vehicles
plot3(Geometry.V1PosStart(1),Geometry.V1PosStart(2), Geometry.V1PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineStyle', 'none' );
text(Geometry.V1PosStart(1),Geometry.V1PosStart(2), Geometry.V1PosStart(3) , 'Tx1')
plt1 = plot3(Geometry.V2PosStart(1),Geometry.V2PosStart(2), Geometry.V2PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'r', 'MarkerSize', 15,'DisplayName', 'Vehicle', 'LineStyle', 'none' );
text(Geometry.V2PosStart(1),Geometry.V2PosStart(2), Geometry.V2PosStart(3) , 'Tx2')


% Plot Trajectories
plt2 = plot3(Geometry.V1Positions(1,:), Geometry.V1Positions(2,:), Geometry.V1Positions(3,:), '-', 'Color','m', 'LineWidth', 2, 'DisplayName', 'TrajectoryV1');
plt3 = plot3(Geometry.V2Positions(1,:), Geometry.V2Positions(2,:), Geometry.V2Positions(3,:), '-', 'Color','m', 'LineWidth', 2, 'DisplayName', 'TrajectoryV2');

% Plot Tracks
trackV1 = [[Geometry.BSPos(1) Geometry.V1PosStart(1)] ; [Geometry.BSPos(2) Geometry.V1PosStart(2)] ; [Geometry.BSPos(3) Geometry.V1PosStart(3)]];
plt4_v1 = plot3(trackV1(1,:), trackV1(2,:), trackV1(3,:), '--', 'Color','k', 'DisplayName', 'TrackV1');
trackV2 = [[Geometry.BSPos(1) Geometry.V2PosStart(1)] ; [Geometry.BSPos(2) Geometry.V2PosStart(2)] ; [Geometry.BSPos(3) Geometry.V2PosStart(3)]];
plt4_v2 = plot3(trackV2(1,:), trackV2(2,:), trackV2(3,:), '--', 'Color','k', 'DisplayName', 'TrackV2');


% Plot ground and sky
X = [-50:220];
Y = [-150:150];
Floor.x = repmat(X(1:end-1),Y(end)-Y(1),1);
Floor.y = repmat(Y(1:end-1),X(end)-X(1),1);
Floor.z = zeros(X(end)-X(1), Y(end)-Y(1));
plt5 = surf(Floor.x', Floor.y, Floor.z,'EdgeColor',[0.9290 0.6940 0.1250]);

Z = [0:26];
Y = [-150:150];
Floor.z = repmat(Z(1:end-1),Y(end)-Y(1),1);
Floor.y = repmat(Y(1:end-1),Z(end)-Z(1),1);
Floor.x = zeros(Z(end)-Z(1), Y(end)-Y(1))-50;
plt6 = surf(Floor.x, Floor.y, Floor.z','EdgeColor',[0.3010 0.7450 0.9330]);

Z = [0:26];
X = [-51:220];
Floor.z = repmat(Z(1:end-1),X(end)-X(1),1);
Floor.x = repmat(X(1:end-1),Z(end)-Z(1),1);
Floor.y = zeros(Z(end)-Z(1), X(end)-X(1))+150;
plt6 = surf(Floor.x, Floor.y, Floor.z','EdgeColor',[0.3010 0.7450 0.9330]);








legend([plt(floor(Geometry.Nant^2/2)) plt1 plt2 plt3 plt4_v1 plt4_v2])

grid on
box on
view(45, 45);

xlim([-50 220])
ylim([-150 150])
zlim([0 25])

xlabel('x-coord in [m]', 'FontSize', 18);
ylabel('y-coord in [m]','FontSize', 18);
zlabel('z-coord in [m]','FontSize', 18);

hold off


