package com.example.loan;

import org.springframework.core.io.ClassPathResource;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

@RestController
public class PredictController {
    private Classifier model;
    private Instances header;

    private synchronized void load() throws Exception {
        if (model != null) return;
        ClassPathResource res = new ClassPathResource("loan.model");
        if (!res.exists()) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "Model not found. Run Train.java first, then restart the app.");
        }
        Object[] objs = SerializationHelper.readAll(res.getInputStream());
        model = (Classifier) objs[0];
        header = (Instances) objs[1];
    }

    /** Lists the input fields so the web page can build its form automatically. */
    @GetMapping("/schema")
    public List<Map<String, Object>> schema() throws Exception {
        load();
        List<Map<String, Object>> fields = new ArrayList<>();
        for (int i = 0; i < header.numAttributes(); i++) {
            if (i == header.classIndex()) continue;
            Attribute a = header.attribute(i);
            Map<String, Object> f = new LinkedHashMap<>();
            f.put("name", a.name());
            f.put("type", a.isNumeric() ? "numeric" : "nominal");
            if (a.isNominal()) f.put("values", Collections.list(a.enumerateValues()));
            fields.add(f);
        }
        return fields;
    }

    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestBody Map<String, String> input) throws Exception {
        load();
        Instance row = new DenseInstance(header.numAttributes()); // every value starts as missing
        row.setDataset(header);
        for (int i = 0; i < header.numAttributes(); i++) {
            if (i == header.classIndex()) continue;
            Attribute a = header.attribute(i);
            String v = input.get(a.name());
            if (v == null || v.isBlank()) continue;               // left missing
            try {
                if (a.isNumeric()) row.setValue(a, Double.parseDouble(v.trim()));
                else if (a.indexOfValue(v) >= 0) row.setValue(a, v);
                else throw new IllegalArgumentException("Unknown value '" + v + "' for " + a.name());
            } catch (NumberFormatException e) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, a.name() + " must be a number");
            } catch (IllegalArgumentException e) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());
            }
        }
        double[] probs = model.distributionForInstance(row);
        int pos = Math.max(0, header.classAttribute().indexOfValue("1"));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("defaultRisk", probs[pos]);
        return out;
    }
}
