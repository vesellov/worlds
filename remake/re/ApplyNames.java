// Ghidra post-script: gives the functions of game.exe the names found by the static analysis
// (re/names.json: address -> Class::method) and writes the decompiled C of every function into one file.
//
//     analyzeHeadless <project dir> EvilIslands -import game.exe -scriptPath remake/re \
//         -postScript ApplyNames.java remake/re/names.json <output dir>
//
// The names file "-" only decompiles: the names are addresses of the first game.exe and do not fit other builds.
//
// Java instead of Python: Ghidra 12 ships Jython only as an optional extension.
//@category EvilIslands

import java.io.FileReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.file.Paths;
import java.util.Map;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.symbol.SourceType;

public class ApplyNames extends GhidraScript {

    private static final int DECOMPILE_TIMEOUT = 60;

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        String namesFilePath = args[0];
        String outputDirPath = args[1];
        if (!namesFilePath.equals("-")) {
            JsonObject names;
            try (Reader reader = new FileReader(namesFilePath)) {
                names = JsonParser.parseReader(reader).getAsJsonObject();
            }
            applyNames(names);
        }
        decompileAll(outputDirPath);
    }

    private void applyNames(JsonObject names) {
        FunctionManager functions = currentProgram.getFunctionManager();
        int named = 0;
        for (Map.Entry<String, JsonElement> entry : names.entrySet()) {
            Address address = toAddr("0x" + entry.getKey());
            String name = entry.getValue().getAsString().replace("::", "__").replace("~", "D_").replace("?", "_q");
            try {
                Function function = functions.getFunctionAt(address);
                if (function == null && !name.endsWith("vftable")) {
                    function = createFunction(address, name);
                }
                if (function != null) {
                    function.setName(name, SourceType.USER_DEFINED);
                } else {
                    createLabel(address, name, true);
                }
                named++;
            } catch (Exception e) {
                println("can not name " + entry.getKey() + " " + name + ": " + e.getMessage());
            }
        }
        println("named " + named + " of " + names.size());
    }

    private void decompileAll(String outputDirPath) throws Exception {
        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);
        int count = 0;
        int failed = 0;
        String outputFilePath = Paths.get(outputDirPath, "game_exe.c").toString();
        try (PrintWriter out = new PrintWriter(outputFilePath, "UTF-8")) {
            FunctionIterator iterator = currentProgram.getFunctionManager().getFunctions(true);
            while (iterator.hasNext() && !monitor.isCancelled()) {
                Function function = iterator.next();
                DecompileResults result = decompiler.decompileFunction(function, DECOMPILE_TIMEOUT, monitor);
                if (result != null && result.decompileCompleted()) {
                    out.println();
                    out.println("//==== " + function.getName() + " @ " + function.getEntryPoint());
                    out.print(result.getDecompiledFunction().getC());
                    count++;
                } else {
                    failed++;
                }
            }
        }
        decompiler.dispose();
        println("decompiled " + count + " functions, failed " + failed + ", written to " + outputFilePath);
    }
}
