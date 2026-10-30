import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DoubleDataBox;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.Fask;
import edu.cmu.tetrad.search.score.SemBicScore;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class FaskCsvRunner {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException("Usage: FaskCsvRunner <input_csv> <output_json> [two_cycle_alpha] [penalty_discount]");
        }

        Path inputCsv = Path.of(args[0]);
        Path outputJson = Path.of(args[1]);
        double twoCycleAlpha = args.length >= 3 ? Double.parseDouble(args[2]) : 0.01;
        double penaltyDiscount = args.length >= 4 ? Double.parseDouble(args[3]) : 2.0;

        double[][] values = readCsv(inputCsv);
        int rows = values.length;
        int cols = values[0].length;

        List<Node> variables = new ArrayList<>();
        for (int i = 0; i < cols; i++) {
            variables.add(new ContinuousVariable("ROI" + (i + 1)));
        }

        DataSet dataSet = new BoxDataSet(new DoubleDataBox(values), variables);
        SemBicScore score = new SemBicScore(dataSet, true);
        score.setPenaltyDiscount(penaltyDiscount);

        Fask fask = new Fask(dataSet, score);
        fask.setTwoCycleAlpha(twoCycleAlpha);
        Graph graph = fask.search();

        boolean[][] mask = new boolean[cols][cols];
        List<String> edgeTexts = new ArrayList<>();
        for (Edge edge : graph.getEdges()) {
            Node n1 = edge.getNode1();
            Node n2 = edge.getNode2();
            int i = variables.indexOf(n1);
            int j = variables.indexOf(n2);
            if (i < 0 || j < 0 || i == j) {
                continue;
            }
            if (edge.pointsTowards(n2)) {
                mask[i][j] = true;
                edgeTexts.add(n1.getName() + "->" + n2.getName());
            }
            if (edge.pointsTowards(n1)) {
                mask[j][i] = true;
                edgeTexts.add(n2.getName() + "->" + n1.getName());
            }
            if (Edges.isBidirectedEdge(edge) || Edges.isUndirectedEdge(edge) || Edges.isNondirectedEdge(edge)) {
                mask[i][j] = true;
                mask[j][i] = true;
                edgeTexts.add(n1.getName() + "<->" + n2.getName());
            }
        }

        Files.writeString(outputJson, toJson(mask, edgeTexts, rows, cols, twoCycleAlpha, penaltyDiscount), StandardCharsets.UTF_8);
    }

    private static double[][] readCsv(Path inputCsv) throws IOException {
        List<double[]> rows = new ArrayList<>();
        int width = -1;
        try (BufferedReader reader = Files.newBufferedReader(inputCsv, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",");
                if (width < 0) {
                    width = parts.length;
                } else if (parts.length != width) {
                    throw new IOException("CSV has inconsistent column count.");
                }
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                rows.add(row);
            }
        }
        if (rows.isEmpty() || width <= 1) {
            throw new IOException("CSV must contain at least one row and two columns.");
        }
        return rows.toArray(new double[0][]);
    }

    private static String toJson(boolean[][] mask, List<String> edges, int rows, int cols, double twoCycleAlpha, double penaltyDiscount) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"rows\":").append(rows).append(",");
        sb.append("\"cols\":").append(cols).append(",");
        sb.append("\"two_cycle_alpha\":").append(twoCycleAlpha).append(",");
        sb.append("\"penalty_discount\":").append(penaltyDiscount).append(",");
        sb.append("\"mask\":[");
        for (int i = 0; i < mask.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("[");
            for (int j = 0; j < mask[i].length; j++) {
                if (j > 0) {
                    sb.append(",");
                }
                sb.append(mask[i][j] ? "true" : "false");
            }
            sb.append("]");
        }
        sb.append("],");
        sb.append("\"edges\":[");
        for (int i = 0; i < edges.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("\"").append(edges.get(i).replace("\\", "\\\\").replace("\"", "\\\"")).append("\"");
        }
        sb.append("]}");
        return sb.toString();
    }
}
