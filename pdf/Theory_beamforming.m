%{

Phase Shift Beamformer:
A beamformer can be considered a spatial filter that suppresses
the signal from all directions, except the desired ones. A conventional
beamformer simply delays the received signal at each antenna so that 
the signals are aligned as if they arrive at all the antennas at the 
same time. In the narrowband case, this is equivalent to multiplying 
the signal received at each antenna by a phase factor. 



Self Nulling Issue in MVDR:
The MVDR beamformer preserves the signal arriving along a desired
direction, while trying to suppress signals coming from other directions.

On many occasions, we may not be able to separate the interference
from the target signal, and therefore, the MVDR beamformer has to
calculate weights using data that includes the target signal. In 
this case, if the target signal is received along a direction slightly
different from the desired one, the MVDR beamformer suppresses it. 
This occurs because the MVDR beamformer treats all the signals, except
the one along the desired direction, as undesired interferences. This
effect is sometimes referred to as "signal self nulling".


LCMV Beamformer:
To prevent signal self-nulling, we can use an LCMV beamformer, which 
allows us to put multiple constraints along the target direction 
(steering vector). It reduces the chance that the target signal will
be suppressed when it arrives at a slightly different angle from the 
desired direction. 



%}
