function fig = PlotScenario_SNR(Geometry)

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
plt11 = plot3(Geometry.V1PosStart(1),Geometry.V1PosStart(2), Geometry.V1PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'r', 'MarkerSize',15,'DisplayName', 'Vehicle1', 'LineStyle', 'none' );
text(Geometry.V1PosStart(1),Geometry.V1PosStart(2), Geometry.V1PosStart(3) , 'Tx1')
plt1 = plot3(Geometry.V2PosStart(1),Geometry.V2PosStart(2), Geometry.V2PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'g', 'MarkerSize', 15,'DisplayName', 'Vehicle2', 'LineStyle', 'none' );
text(Geometry.V2PosStart(1),Geometry.V2PosStart(2), Geometry.V2PosStart(3) , 'Tx2')
plt2 = plot3(Geometry.V3PosStart(1),Geometry.V3PosStart(2), Geometry.V3PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'b', 'MarkerSize', 15,'DisplayName', 'Vehicle3', 'LineStyle', 'none' );
text(Geometry.V3PosStart(1),Geometry.V3PosStart(2), Geometry.V3PosStart(3) , 'Tx3')
plt3 = plot3(Geometry.V4PosStart(1),Geometry.V4PosStart(2), Geometry.V4PosStart(3) , 'Marker', 'x', 'MarkerEdgeColor', 'r','MarkerFaceColor', 'y', 'MarkerSize', 15,'DisplayName', 'Vehicle4', 'LineStyle', 'none' );
text(Geometry.V4PosStart(1),Geometry.V4PosStart(2), Geometry.V4PosStart(3) , 'Tx4')


% Plot Tracks
trackV1 = [[Geometry.BSPos(1) Geometry.V1PosStart(1)] ; [Geometry.BSPos(2) Geometry.V1PosStart(2)] ; [Geometry.BSPos(3) Geometry.V1PosStart(3)]];
plt4_v1 = plot3(trackV1(1,:), trackV1(2,:), trackV1(3,:), '--', 'Color','k', 'DisplayName', 'TrackV1');
trackV2 = [[Geometry.BSPos(1) Geometry.V2PosStart(1)] ; [Geometry.BSPos(2) Geometry.V2PosStart(2)] ; [Geometry.BSPos(3) Geometry.V2PosStart(3)]];
plt4_v2 = plot3(trackV2(1,:), trackV2(2,:), trackV2(3,:), '--', 'Color','k', 'DisplayName', 'TrackV2');
trackV3 = [[Geometry.BSPos(1) Geometry.V3PosStart(1)] ; [Geometry.BSPos(2) Geometry.V3PosStart(2)] ; [Geometry.BSPos(3) Geometry.V3PosStart(3)]];
plt4_v3 = plot3(trackV3(1,:), trackV3(2,:), trackV3(3,:), '--', 'Color','k', 'DisplayName', 'TrackV3');
trackV4 = [[Geometry.BSPos(1) Geometry.V4PosStart(1)] ; [Geometry.BSPos(2) Geometry.V4PosStart(2)] ; [Geometry.BSPos(3) Geometry.V4PosStart(3)]];
plt4_v4 = plot3(trackV4(1,:), trackV4(2,:), trackV4(3,:), '--', 'Color','k', 'DisplayName', 'TrackV4');


% Plot ground and sky
X = [-50:50];
Y = [-50:50];
Floor.x = repmat(X(1:end-1),Y(end)-Y(1),1);
Floor.y = repmat(Y(1:end-1),X(end)-X(1),1);
Floor.z = zeros(X(end)-X(1), Y(end)-Y(1))-1;
% plt5 = surf(Floor.x', Floor.y, Floor.z,'EdgeColor',	[1 1 1]);

Z = [-1:26];
Y = [-51:50];
Floor.z = repmat(Z(1:end-1),Y(end)-Y(1),1);
Floor.y = repmat(Y(1:end-1),Z(end)-Z(1),1);
Floor.x = zeros(Z(end)-Z(1), Y(end)-Y(1))-50;
plt6 = surf(Floor.x, Floor.y, Floor.z','EdgeColor',[0.3010 0.7450 0.9330]);

Z = [-1:26];
X = [-51:51];
Floor.z = repmat(Z(1:end-1),X(end)-X(1),1);
Floor.x = repmat(X(1:end-1),Z(end)-Z(1),1);
Floor.y = zeros(Z(end)-Z(1), X(end)-X(1))+49;
plt6 = surf(Floor.x, Floor.y, Floor.z','EdgeColor',[0.3010 0.7450 0.9330]);



legend([plt(floor(Geometry.Nant^2/2)) plt11 plt1 plt2 plt3 plt4_v1 plt4_v2 plt4_v3 plt4_v4])

grid on
box on
view(0, 90);

xlim([-50 50])
ylim([-33 33])
zlim([-1 25])

xlabel('x-coord in [m]', 'FontSize', 18);
ylabel('y-coord in [m]','FontSize', 18);
zlabel('z-coord in [m]','FontSize', 18);

hold off



set(0,'DefaultTextFontSize',18)
set(0,'DefaultLineLineWidth',2);
%set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)

channel = PlotScenario_SNR(Geometry);
saveas(channel,'Scenario_cirle.png')
