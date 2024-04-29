package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;



// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.utils.Coordinate;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double INITIAL_EXPLORATION_PROB = 1.0;
    public static final double FINAL_EXPLORATION_PROB = 0.005;
    public static final double EXPLORATION_DECAY_RATE = 0.000001;


    private Random random;
    private long movesMade = 0;
    private long gameMoves = 0;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    
    }
    private Map<String, Map<String, Integer>> visitCountMap = new HashMap<>();

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int numInputs = Board.NUM_ROWS * Board.NUM_COLS + 7 + 4 + 2;  // Adjusted for additional features
        final int hiddenDim1 = 300; // Example: Half of the input size, but not less than 150
        final int hiddenDim2 = 150;                // A smaller second hidden layer
        final int outputDim = 1;                              // Output layer size remains 1
    
        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numInputs, hiddenDim1));      // First hidden layer
        qFunction.add(new ReLU());                            // Activation for first hidden layer
        qFunction.add(new Dense(hiddenDim1, hiddenDim2));     // Second hidden layer
        qFunction.add(new ReLU());                            // Activation for second hidden layer
        qFunction.add(new Dense(hiddenDim2, outputDim));      

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction){
        Matrix flattenedImage = null;
    try {
        // This gets the board state as a flat Matrix (1D)
        flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
    } catch(Exception e) {
        e.printStackTrace();
        System.exit(-1);
    }
    int originalLength = flattenedImage.numel();  
    int extraFeatures = Mino.MinoType.values().length + 4 + 2;  // Total extra features (num Mino types + 4 orientations + 2 position values)
    
    
    Matrix extendedMatrix = Matrix.full(1, originalLength + extraFeatures, 0);

    int numMinoTypes = Mino.MinoType.values().length;
    int numOrientations = 4; // Assuming there are 4 orientations

    for (int i = 0; i < originalLength; i++) {
        extendedMatrix.set(0, i, flattenedImage.get(0, i));
    }

    int index = originalLength; // Start index for additional features

    for (int i = 0; i < Mino.MinoType.values().length; i++) {
        extendedMatrix.set(0, index++, i == potentialAction.getType().ordinal() ? 1 : 0);
    }
    
    for (int i = 0; i < 4; i++) {  // Assuming four orientations: A, B, C, D
        extendedMatrix.set(0, index++, i == potentialAction.getOrientation().ordinal() ? 1 : 0);
    }
    
    Coordinate position = potentialAction.getPivotBlockCoordinate();
    extendedMatrix.set(0, index++, position.getXCoordinate());
    extendedMatrix.set(0, index, position.getYCoordinate());

     //System.out.println("Final Extended Matrix with All Features:");
     //for (int i = 0; i < extendedMatrix.numel(); i++) {
     //    System.out.print(extendedMatrix.get(0, i) + " ");
     //    if ((i + 1) % 10 == 0) System.out.println();  // New line every 10 elements for readability
     //}
     //System.out.println(); // Final new line for separation

    return extendedMatrix;
}
    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */

    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        //if (game.didAgentLose()){
        //    gameMoves = 0;
        //}
        movesMade++;
        //gameMoves++;
        if (INITIAL_EXPLORATION_PROB - EXPLORATION_DECAY_RATE * (movesMade - 1) < FINAL_EXPLORATION_PROB) {
            return this.getRandom().nextDouble() <= FINAL_EXPLORATION_PROB;
        }

        else {
            return this.getRandom().nextDouble() <= INITIAL_EXPLORATION_PROB - EXPLORATION_DECAY_RATE * (movesMade - 1);
        }
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        if (possibleMoves.isEmpty()) {
            return null; // No moves available, return null
        }
    
        Mino selectedMove = null;
        double highestExplorationValue = Double.NEGATIVE_INFINITY;
    
        for (Mino move : possibleMoves) {
            String stateKey = gameToString(game,move);
            String actionKey = actionToString(move);
    
            double qValue = getQValue(game, move); // Assume getQValue is defined elsewhere
            int visitCount = getVisitCount(stateKey, actionKey);
    
            double explorationValue = qValue + (1.0 / Math.max(1, visitCount));
    
            if (explorationValue > highestExplorationValue) {
                highestExplorationValue = explorationValue;
                selectedMove = move;
            }
        }
    
        if (selectedMove != null) {
            String stateKey = gameToString(game,selectedMove);
            String actionKey = actionToString(selectedMove);
            updateVisitCount(stateKey, actionKey);
        }
    
        return selectedMove;
    }

    public double getQValue(GameView game, Mino action) {
        // Preprocess the game state and the action into an input vector for the neural network
        Matrix inputVector = getQFunctionInput(game, action);
        Matrix qValue = null;        // Pass the input vector through the network to get the Q-value
        try {
            qValue = this.getQFunction().forward(inputVector); // This method might throw an exception
        } catch (Exception e) {
            System.err.println("Failed to forward input vector: " + e.getMessage());
            // Handle the error or rethrow as unchecked
            throw new RuntimeException("Error during network forward pass", e);
        }
        // Return the Q-value
        return qValue.get(0, 0);
    }

    private String gameToString(final GameView game, final Mino potentialAction) {
        Matrix state = null;
        try {
            state = game.getGrayscaleImage(potentialAction);  // This method might throw an exception
        } catch (Exception e) {
            System.err.println("Failed to get grayscale image: " + e.getMessage());
            // Handle the error or rethrow as unchecked if you cannot recover here
            throw new RuntimeException("Error processing grayscale image", e);
        }  // Assuming getGrayscaleImage returns a Matrix
        StringBuilder sb = new StringBuilder();
        int rows = Board.NUM_ROWS;
        int cols = Board.NUM_COLS;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.append(state.get(i, j) > 0 ? "1" : "0");
            }
        }
        return sb.toString();
    }
    
    private String actionToString(final Mino potentialAction) {
        // Include the type and orientation of the Mino, and optionally the position
        return potentialAction.getType().toString() + "_" +
               potentialAction.getOrientation().toString();
    }

    private void updateVisitCount(String stateKey, String actionKey) {
        visitCountMap.putIfAbsent(stateKey, new HashMap<>());
        Map<String, Integer> actionMap = visitCountMap.get(stateKey);
        actionMap.put(actionKey, actionMap.getOrDefault(actionKey, 0) + 1);
    }
    
    private int getVisitCount(String stateKey, String actionKey) {
        return visitCountMap.getOrDefault(stateKey, new HashMap<>())
                            .getOrDefault(actionKey, 0);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        Board board = game.getBoard();
    
        // Initialize reward components
        double reward = 0.0;
        int stackHeight = calculateStackHeight(board, game);
        int numHoles = calculateHoles(board, game);
        int numCompleteLines = calculateNumCompleteLines(board, game);
        double flatness = flatTopReward(board, game);
        int isNinetyStack = ten_stack(board, game);
    
        // Reward for clearing lines and encouraging progress
        reward += numCompleteLines * 100000; // Encourage clearing lines
        if (numCompleteLines > 0){
            System.out.println("yes");
            System.out.println(numCompleteLines);
        }
        ///System.out.println(reward);
        reward += game.getScoreThisTurn() * 100000; // Reward based on current score
        //System.out.println(reward);
    
        // Penalty for high stack height
        if (stackHeight > 4) {
            reward -= Math.pow(stackHeight, 2) * 5; // Penalize high stack heights
            //System.out.println(reward);
        }
    
        // Penalty for creating holes
        reward -= numHoles * 10;
        //System.out.println(reward);
    
        // Reward for maintaining flatness or avoiding flat tops
        reward += flatness; // Reward for maintaining flatness
        //System.out.println(reward);
        reward += isNinetyStack * 50; // Reward for creating a "90-stack"
        //System.out.println(reward);
    
        // Sparse rewards
        //if (numCompleteLines >= 2) {
        //    reward += 100000; // Large reward for clearing a Tetris (4 lines)
        //}
    
        for(int i = 0; i < board.NUM_COLS; i++) {
            // Penalty for losing (if the top row is occupied, we lose
        if ((board.isCoordinateOccupied(i, 0))) {
            reward -= 100000; // Big penalty for losing
        }
        }


        // Normalize reward
        return reward / 10000;
    }

    private int calculateStackHeight(Board board, GameView game) {
        // implementation to calculate stack height
        //System.out.println(board.NUM_ROWS); -> 22
        //System.out.println(board.NUM_COLS); -> 10
        for (int i = 0; i < board.NUM_ROWS; i++) {
            for (int j = 0; j < board.NUM_COLS; j++) {
                if (board.isCoordinateOccupied(j, i)) {
                    return board.NUM_ROWS - i; // Could be wrong here.
                }
            }
        }
        return 0;
    }
    
    
    private int calculateNumCompleteLines(Board board, GameView game) {
        // implementation to calculate number of complete lines
        int num_complete_lines = 0;
        boolean checker = true;

        for (int i = 0; i < board.NUM_ROWS; i++) {
            for (int j = 0; j < board.NUM_COLS; j++) {
                if (!board.isCoordinateOccupied(j, i)) {
                    checker = false;
                    break;
                }
            }
            if (checker == true) {
                num_complete_lines++;
            }
            checker = true;
        }
        return num_complete_lines;
    }
    
    private int calculateHoles(Board board, GameView game) {
        int holes = 0;
        for (int col = 0; col < board.NUM_COLS; col++) {
            boolean foundBlock = false;
            for (int row = 0; row < board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    foundBlock = true;
                } else if (foundBlock) {
                    holes++;
                }
            }
        }
        return holes;
    }

    private boolean rest_row_full(Board board, GameView game, int row, int column) {
        //Checks if a single cell is empty in a row. Returns true or false.
        boolean checker = true;
        for (int col = 0; col < board.NUM_COLS; col++) {
            if (col == column) {
                continue;
            }
            else if (!board.isCoordinateOccupied(col, row)){
                checker = false;
                break;
            }
        }
        return checker;
    }

    private double flatTopReward(Board board, GameView game) {
        int playfieldHeight = board.NUM_ROWS;
        int lowerRegionHeight = playfieldHeight / 2; // Middle point of the playfield
        double rewardMultiplier = 2.0; // Reward multiplier for flatness
        
        double reward = 0.0;
        int currentHeight = calculateStackHeight(board, game);
        int flatness = calculateFlatness(board);
        
        // If the current height is below the middle point, penalize flat stacks
        if (currentHeight < lowerRegionHeight) {
            // Calculate penalty based on flatness
            // System.out.println(flatness);
            double flatnessPenalty = rewardMultiplier / (flatness != 0 ? flatness : 1); // Avoid division by zero
            reward += flatnessPenalty;
        }
        // If the current height is at or above the middle point, reward maintaining a flat top
        else {
            // Calculate reward based on flatness
            double flatnessReward = (flatness * rewardMultiplier) / 2;
            reward -= flatnessReward;
        }
        
        return reward;
    }    
    
    private int calculateFlatness(Board board) {
        int[] columnHeights = calculateColumnHeights(board);
        if (columnHeights.length == 0) {
            return 0;
        }
        int flatness = calculateStandardDeviation(columnHeights);
        return flatness;
    }
    
    private int calculateStandardDeviation(int[] values) {
        if (values.length == 0) {
            return 0;
        }
    
        double sum = 0.0;
        double sumSquareDiff = 0.0;
        int n = values.length;
        
        // Calculate mean
        for (int value : values) {
            sum += value;
        }
        double mean = sum / n;
        
        // Calculate sum of squared differences
        for (int value : values) {
            double diff = value - mean;
            sumSquareDiff += diff * diff;
        }
        
        // Calculate standard deviation
        double variance = sumSquareDiff / n;
        double standardDeviation = Math.sqrt(variance);
        
        return (int) Math.round(standardDeviation);
    }
    
    private int[] calculateColumnHeights(Board board) {
        int[] columnHeights = new int[board.NUM_COLS-1];
        
        for (int col = 0; col < board.NUM_COLS-1; col++) {
            int height = board.NUM_ROWS;
            for (int row = board.NUM_ROWS - 1; row >= 0; row--) {
                if (board.isCoordinateOccupied(col, row)) {
                    break;
                }
                height--;
            }
            columnHeights[col] = height;
        }
        return columnHeights;
    }
    
    private int ten_stack(Board board, GameView game) {
        int reward = 0;
        boolean checker = true;
        
        if (empty_column(board, game, board.NUM_COLS-1)) {
            reward += 20;
        }

        return reward;
    }

    private boolean empty_column(Board board, GameView game, int col) {
        boolean checker = true;

        for (int row = 0; row < board.NUM_ROWS; row++) {
            if (board.isCoordinateOccupied(col, row)) {
                checker = false;
            }
        }

        return checker;
    }
    

}