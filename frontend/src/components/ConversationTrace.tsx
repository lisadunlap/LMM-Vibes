import React from "react";
import type { Message } from "../lib/traces";
import { Box, Typography } from "@mui/material";

export function ConversationTrace({ messages }: { messages: Message[] }) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {messages.map((m, i) => (
        <Box key={i} sx={{
          p: 1.5,
          border: "1px solid #e5e7eb",
          borderRadius: 1,
          backgroundColor: m.role === "user" ? "#f8fafc" : "#ffffff",
        }}>
          <Typography variant="caption" color="text.secondary">
            {m.role}
          </Typography>
          <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
            {String(m.content ?? "")}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

export default ConversationTrace;


