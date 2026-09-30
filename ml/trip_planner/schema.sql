-- Trip planner tables (TRIP_PLANNER_DESIGN.md §14). Idempotent: run on every start.
-- The ML service writes only the ml schema; Spring owns business tables (users, clans, expenses).
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.trips (
    trip_id         text PRIMARY KEY,
    event           jsonb NOT NULL,              -- snapshot of the event (times drive the timeline)
    clan_id         bigint,
    created_by      bigint,
    status          text NOT NULL DEFAULT 'active',
    prefs           jsonb NOT NULL,
    items           jsonb NOT NULL,
    locks           jsonb NOT NULL DEFAULT '{}',
    memory          text NOT NULL DEFAULT '',
    memory_upto     bigint NOT NULL DEFAULT 0,
    current_version int NOT NULL,
    parent_version  int,
    busy            boolean NOT NULL DEFAULT false,   -- a worker is running a job for this trip
    busy_since      timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml.trip_members (
    trip_id     text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    user_id     bigint NOT NULL,
    first_name  text NOT NULL DEFAULT '',
    origin_city text NOT NULL DEFAULT '',
    role        text NOT NULL DEFAULT 'member',
    joined_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (trip_id, user_id)
);
CREATE INDEX IF NOT EXISTS trip_members_user ON ml.trip_members (user_id);

CREATE TABLE IF NOT EXISTS ml.trip_versions (      -- every version, kept forever
    trip_id    text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    version    int NOT NULL,
    parent     int,
    items      jsonb NOT NULL,
    prefs      jsonb NOT NULL,
    locks      jsonb NOT NULL,
    author     text NOT NULL,
    reason     text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (trip_id, version)
);

CREATE TABLE IF NOT EXISTS ml.trip_messages (      -- the chat, kept forever (erasure on request)
    seq        bigserial PRIMARY KEY,
    trip_id    text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    role       text NOT NULL,                        -- user | assistant
    sender_id  bigint,
    content    text NOT NULL,
    model      text,
    tokens_in  int,
    tokens_out int,
    latency_ms int,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trip_messages_trip ON ml.trip_messages (trip_id, seq);
CREATE INDEX IF NOT EXISTS trip_messages_sender ON ml.trip_messages (sender_id, created_at);

CREATE TABLE IF NOT EXISTS ml.trip_jobs (          -- the work queue (FOR UPDATE SKIP LOCKED), no Redis
    job_id      text PRIMARY KEY,
    trip_id     text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    kind        text NOT NULL,                       -- draft | chat
    payload     jsonb NOT NULL DEFAULT '{}',
    status      text NOT NULL DEFAULT 'queued',      -- queued | running | done | failed
    attempts    int NOT NULL DEFAULT 0,
    run_after   timestamptz NOT NULL DEFAULT now(),
    error       text,
    created_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
    started_at  timestamptz,
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS trip_jobs_queue ON ml.trip_jobs (status, run_after, created_at);
CREATE INDEX IF NOT EXISTS trip_jobs_trip ON ml.trip_jobs (trip_id, created_at);

CREATE TABLE IF NOT EXISTS ml.trip_events (        -- what the SSE stream sends; also a durable activity log
    seq     bigserial PRIMARY KEY,
    trip_id text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    job_id  text,
    type    text NOT NULL,                           -- status | tool | message | trip_updated | done | error
    data    jsonb NOT NULL DEFAULT '{}',
    at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trip_events_trip ON ml.trip_events (trip_id, seq);

-- groups (design §9): variants on the trip row, item votes, group decisions and their votes
ALTER TABLE ml.trips ADD COLUMN IF NOT EXISTS variants jsonb NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS ml.trip_votes (         -- thumbs up / down per item, one per member
    trip_id    text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    item_id    text NOT NULL,
    user_id    bigint NOT NULL,
    value      smallint NOT NULL CHECK (value IN (-1, 1)),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (trip_id, item_id, user_id)
);

CREATE TABLE IF NOT EXISTS ml.trip_decisions (
    decision_id text PRIMARY KEY,
    trip_id     text NOT NULL REFERENCES ml.trips ON DELETE CASCADE,
    kind        text NOT NULL,                       -- variant | swap | lock | unlock | pref
    payload     jsonb NOT NULL DEFAULT '{}',
    options     jsonb NOT NULL,
    question    text NOT NULL,
    opened_by   bigint,                              -- null: opened by the planner (clan draft variants)
    deadline    timestamptz NOT NULL,
    status      text NOT NULL DEFAULT 'open',        -- open | passed | rejected | tied | expired | failed
    result      jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);
CREATE INDEX IF NOT EXISTS trip_decisions_open ON ml.trip_decisions (status, deadline);
CREATE INDEX IF NOT EXISTS trip_decisions_trip ON ml.trip_decisions (trip_id, created_at);

CREATE TABLE IF NOT EXISTS ml.decision_votes (
    decision_id text NOT NULL REFERENCES ml.trip_decisions ON DELETE CASCADE,
    user_id     bigint NOT NULL,
    option      text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (decision_id, user_id)
);

-- draft cache (design §4.1): the LLM's plan per (event, preferences, group size); origin cities don't change it
CREATE TABLE IF NOT EXISTS ml.draft_cache (
    key        text PRIMARY KEY,
    event_id   bigint,
    plan       jsonb NOT NULL,
    model      text,
    hits       int NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    used_at    timestamptz
);

CREATE TABLE IF NOT EXISTS ml.link_clicks (       -- hand-off clicks: conversion tracking with the partners' subid
    id         bigserial PRIMARY KEY,
    trip_id    text,
    item_id    text,
    user_id    bigint,
    provider   text,
    host       text,
    clicked_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS link_clicks_trip ON ml.link_clicks (trip_id, clicked_at);

CREATE TABLE IF NOT EXISTS ml.planner_log (        -- validator failures, fallbacks, errors: grows the eval set
    id      bigserial PRIMARY KEY,
    trip_id text,
    job_id  text,
    kind    text NOT NULL,
    detail  jsonb NOT NULL DEFAULT '{}',
    at      timestamptz NOT NULL DEFAULT now()
);
