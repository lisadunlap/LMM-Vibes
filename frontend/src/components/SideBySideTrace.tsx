import React from "react";
import type { Message } from "../lib/traces";
import { Box, Typography } from "@mui/material";
import ConversationTrace from "./ConversationTrace";

export function SideBySideTrace({
  messagesA,
  messagesB,
  modelA,
  modelB,
}: {
  messagesA: Message[];
  messagesB: Message[];
  modelA: string;
  modelB: string;
}) {
  return (
    <Box sx={{
      display: "grid",
      gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
      gap: 2,
    }}>
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{modelA}</Typography>
        <ConversationTrace messages={messagesA} />
      </Box>
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{modelB}</Typography>
        <ConversationTrace messages={messagesB} />
      </Box>
    </Box>
  );
}

export default SideBySideTrace;


