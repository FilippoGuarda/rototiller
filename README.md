# extended_SPADES + ROTOTILLER
a ROS2 compatible implementation of extended SPADES framework
includes the rolling_CHOMP implementation for the ROTOTILLER approach

The original multi_CHOMP implementation is now called multi_chomp_original
While multi_chomp is rolling Chomp

# rolling_CHOMP

ROTOTILLER uses a rolling window implementation of rolling chomp to reduce computation time

# task allocation

ROTOTILLER uses a topological graph of the occupancy grid instead of the full metric map for faster cost compotation, then the allocation is done using google ORTOOLS 
