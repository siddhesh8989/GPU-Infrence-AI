import { pgTable, text, serial, real, integer, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const inferencesTable = pgTable("inferences", {
  id: serial("id").primaryKey(),
  fileId: text("file_id").notNull(),
  filename: text("filename"),
  model: text("model").notNull(),
  computeMode: text("compute_mode").notNull(),
  batchSize: integer("batch_size").notNull().default(1),
  precision: text("precision").notNull().default("fp32"),
  prediction: text("prediction").notNull(),
  confidence: real("confidence").notNull(),
  topPredictions: jsonb("top_predictions"),
  timings: jsonb("timings"),
  memoryUsage: jsonb("memory_usage"),
  pipelineStages: jsonb("pipeline_stages"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

export const insertInferenceSchema = createInsertSchema(inferencesTable).omit({
  id: true,
  createdAt: true,
});

export type InsertInference = z.infer<typeof insertInferenceSchema>;
export type Inference = typeof inferencesTable.$inferSelect;
